"""
OpenAI Responses API Router

Exposes the OpenAI Responses API endpoint (/v1/responses) that internally
converts to/from the Chat Completions format used by CodeBuddy.

This is the third API format supported by codebuddy2api, alongside:
  - OpenAI Chat Completions  (/codebuddy/v1/chat/completions)
  - Anthropic Messages       (/v1/messages)
  - OpenAI Responses         (/v1/responses)   <-- this file

The Responses API is used by tools like Codex CLI.
"""
import json
import time
import uuid
import logging
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse

from .auth import authenticate
from .codebuddy_api_client import codebuddy_api_client
from .codebuddy_token_manager import codebuddy_token_manager
from .usage_stats_manager import usage_stats_manager
from .models_manager import models_manager
from .responses_converter import (
    convert_responses_request_to_chat_completions,
    convert_chat_completion_to_responses,
    ResponsesStreamConverter,
    responses_error_response,
)

# Reuse the stream service and helpers from the codebuddy router
from .codebuddy_router import (
    CodeBuddyStreamService,
    CredentialManager,
    RequestProcessor,
    SSE_HEADERS,
    parse_sse_line,
    get_codebuddy_api_url,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /v1/responses  — OpenAI Responses API
# ---------------------------------------------------------------------------

@router.post("/v1/responses")
async def create_response(
    request: Request,
):
    """OpenAI Responses API endpoint.

    Converts the incoming Responses API request to Chat Completions format,
    forwards it to the CodeBuddy backend, and converts the response back.
    """
    # Authenticate using standard Bearer token
    from .auth import authenticate as auth_func
    from fastapi.security import HTTPBearer

    # Manual auth: Responses API uses Authorization: Bearer (standard OpenAI auth)
    from config import get_server_password
    password = get_server_password()
    if not password:
        raise HTTPException(
            status_code=500,
            detail="CODEBUDDY_PASSWORD is not configured on the server.",
        )

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    else:
        api_key = request.headers.get("x-api-key")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide Authorization: Bearer or x-api-key header.",
        )

    if api_key != password:
        raise HTTPException(status_code=403, detail="Invalid API key")

    try:
        try:
            request_body = await request.json()
        except Exception as e:
            logger.error(f"解析请求体失败: {e}")
            error_body = responses_error_response(400, f"Invalid JSON request body: {str(e)}")
            return JSONResponse(status_code=400, content=error_body)

        # Validate required fields
        if not request_body.get("model"):
            error_body = responses_error_response(400, "model is required")
            return JSONResponse(status_code=400, content=error_body)

        # Convert Responses request -> Chat Completions request
        chat_body = convert_responses_request_to_chat_completions(request_body)
        model = request_body.get("model", "auto-chat")
        client_wants_stream = request_body.get("stream", False)

        # Validate converted request
        try:
            RequestProcessor.validate_request(chat_body)
        except HTTPException as e:
            error_body = responses_error_response(e.status_code, e.detail)
            return JSONResponse(status_code=e.status_code, content=error_body)

        # Get credential
        credential = CredentialManager.get_valid_credential()

        # Generate headers
        headers = codebuddy_api_client.generate_codebuddy_headers(
            bearer_token=credential.get("bearer_token"),
            user_id=credential.get("user_id"),
        )

        # Prepare payload (forces stream=True for CodeBuddy backend)
        payload = RequestProcessor.prepare_payload(chat_body)
        usage_stats_manager.record_model_usage(model)

        service = CodeBuddyStreamService()

        if client_wants_stream:
            return await _handle_responses_stream(
                service, payload, headers, model, request_body
            )
        else:
            return await _handle_responses_non_stream(
                service, payload, headers, model, request_body
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Responses /v1/responses 错误: {e}", exc_info=True)
        error_body = responses_error_response(500, f"Internal server error: {str(e)}")
        return JSONResponse(status_code=500, content=error_body)


async def _handle_responses_non_stream(
    service: CodeBuddyStreamService,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
    original_request: Dict[str, Any],
) -> JSONResponse:
    """Handle non-streaming Responses API response."""
    try:
        chat_response = await service.handle_non_stream_response(payload, headers)
        responses_response = convert_chat_completion_to_responses(
            chat_response, model, original_request
        )
        return JSONResponse(content=responses_response)
    except HTTPException as e:
        error_body = responses_error_response(e.status_code, str(e.detail))
        return JSONResponse(status_code=e.status_code, content=error_body)


async def _handle_responses_stream(
    service: CodeBuddyStreamService,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
    original_request: Dict[str, Any],
) -> StreamingResponse:
    """Handle streaming Responses API response.

    Reuses the CodeBuddy stream service (OpenAI SSE) and converts each chunk
    to Responses API SSE events on the fly.
    """
    converter = ResponsesStreamConverter(model, original_request)

    async def responses_stream():
        try:
            from .codebuddy_router import get_http_client

            client = await get_http_client()
            async with client.stream(
                "POST", get_codebuddy_api_url(), json=payload, headers=headers
            ) as response:
                if response.status_code != 200:
                    error_bytes = await response.aread()
                    error_msg = error_bytes.decode("utf-8", errors="ignore")
                    error_event = _responses_error_sse(
                        f"CodeBuddy API error: {response.status_code} - {error_msg}"
                    )
                    yield error_event
                    return

                buffer = ""
                done = False

                async for chunk in response.aiter_text(chunk_size=8192):
                    if not chunk or done:
                        continue

                    buffer += chunk

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)

                        if not line.strip() or line.startswith(":"):
                            continue

                        if "[DONE]" in line:
                            done = True
                            break

                        chunk_data = parse_sse_line(line)
                        if chunk_data:
                            converted = converter.convert_chunk(chunk_data)
                            if converted:
                                yield converted

                # Process remaining buffer
                if buffer.strip() and not done:
                    chunk_data = parse_sse_line(buffer.strip())
                    if chunk_data:
                        converted = converter.convert_chunk(chunk_data)
                        if converted:
                            yield converted

                # Emit final response.completed event
                yield converter.finalize()

        except Exception as e:
            logger.error(f"Responses stream error: {e}", exc_info=True)
            yield _responses_error_sse(str(e))

    return StreamingResponse(
        responses_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _responses_error_sse(message: str) -> str:
    """Format a Responses API-style error SSE event."""
    event_data = {
        "type": "error",
        "error": {
            "code": "stream_error",
            "message": message,
        },
    }
    return f"event: error\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# GET /v1/models  — Models list (Responses format, same as OpenAI)
# ---------------------------------------------------------------------------

@router.get("/v1/models")
async def list_models(request: Request):
    """List models (same format as OpenAI)."""
    from config import get_server_password
    password = get_server_password()
    if not password:
        raise HTTPException(status_code=500, detail="CODEBUDDY_PASSWORD is not configured.")

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    else:
        api_key = request.headers.get("x-api-key")

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")
    if api_key != password:
        raise HTTPException(status_code=403, detail="Invalid API key")

    try:
        models = models_manager.get_available_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "codebuddy",
                }
                for m in models
            ],
        }
    except Exception as e:
        logger.error(f"获取模型列表错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
