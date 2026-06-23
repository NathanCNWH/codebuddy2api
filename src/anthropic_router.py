"""
Anthropic-compatible API Router

Exposes Anthropic Messages API endpoints (/v1/messages, /v1/models)
that internally convert to/from the OpenAI format used by CodeBuddy.

This mirrors the multi-format approach used by cockpit-tools/CLIProxyAPI:
the same backend (CodeBuddy) is exposed through multiple API formats.
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
from .keyword_replacer import apply_keyword_replacement_to_system_message
from .models_manager import models_manager
from .anthropic_converter import (
    convert_anthropic_request_to_openai,
    convert_openai_response_to_anthropic,
    convert_openai_models_to_anthropic,
    anthropic_error_response,
    AnthropicStreamConverter,
)

# Reuse the stream service and helpers from the codebuddy router
from .codebuddy_router import (
    CodeBuddyStreamService,
    CredentialManager,
    RequestProcessor,
    SSE_HEADERS,
    parse_sse_line,
    format_sse_error,
    get_codebuddy_api_url,
    get_available_models_list,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_anthropic_api_key(request: Request) -> Optional[str]:
    """Extract API key from Anthropic-style header (x-api-key)
    or fall back to Authorization Bearer.
    """
    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return x_api_key

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None


def _anthropic_auth(request: Request) -> str:
    """Authenticate using Anthropic-style headers.

    Anthropic SDK sends `x-api-key` header instead of `Authorization: Bearer`.
    We accept both and validate against the same CODEBUDDY_PASSWORD.
    """
    from config import get_server_password

    password = get_server_password()
    if not password:
        raise HTTPException(
            status_code=500,
            detail="CODEBUDDY_PASSWORD is not configured on the server.",
        )

    api_key = _get_anthropic_api_key(request)
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide x-api-key or Authorization header.",
        )

    if api_key != password:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


# ---------------------------------------------------------------------------
# POST /v1/messages  — Anthropic Messages API
# ---------------------------------------------------------------------------

@router.post("/v1/messages")
async def create_message(
    request: Request,
    anthropic_version: Optional[str] = Header(None, alias="anthropic-version"),
):
    """Anthropic Messages API endpoint.

    Converts the incoming Anthropic request to OpenAI format, forwards it
    to the CodeBuddy backend, and converts the response back to Anthropic format.
    """
    # Authenticate
    _ = _anthropic_auth(request)

    try:
        try:
            request_body = await request.json()
        except Exception as e:
            logger.error(f"解析请求体失败: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON request body: {str(e)}",
            )

        # Validate required Anthropic fields
        if not request_body.get("messages"):
            error_body = anthropic_error_response(
                400, "invalid_request_error", "messages is required"
            )
            return JSONResponse(status_code=400, content=error_body)

        if not request_body.get("max_tokens"):
            error_body = anthropic_error_response(
                400, "invalid_request_error", "max_tokens is required"
            )
            return JSONResponse(status_code=400, content=error_body)

        # Convert Anthropic request -> OpenAI request
        openai_body = convert_anthropic_request_to_openai(request_body)
        model = request_body.get("model", "auto-chat")
        client_wants_stream = request_body.get("stream", False)

        # Validate converted request
        RequestProcessor.validate_request(openai_body)

        # Get credential
        credential = CredentialManager.get_valid_credential()

        # Generate headers
        headers = codebuddy_api_client.generate_codebuddy_headers(
            bearer_token=credential.get("bearer_token"),
            user_id=credential.get("user_id"),
        )

        # Prepare payload (forces stream=True for CodeBuddy backend)
        payload = RequestProcessor.prepare_payload(openai_body)
        usage_stats_manager.record_model_usage(model)

        service = CodeBuddyStreamService()

        if client_wants_stream:
            return await _handle_anthropic_stream(service, payload, headers, model)
        else:
            return await _handle_anthropic_non_stream(
                service, payload, headers, model
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anthropic /v1/messages 错误: {e}", exc_info=True)
        error_body = anthropic_error_response(
            500, "api_error", f"Internal server error: {str(e)}"
        )
        return JSONResponse(status_code=500, content=error_body)


async def _handle_anthropic_non_stream(
    service: CodeBuddyStreamService,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
) -> JSONResponse:
    """Handle non-streaming Anthropic response."""
    try:
        openai_response = await service.handle_non_stream_response(payload, headers)
        anthropic_response = convert_openai_response_to_anthropic(
            openai_response, model
        )
        return JSONResponse(content=anthropic_response)
    except HTTPException as e:
        error_body = anthropic_error_response(
            e.status_code, "api_error", e.detail
        )
        return JSONResponse(status_code=e.status_code, content=error_body)


async def _handle_anthropic_stream(
    service: CodeBuddyStreamService,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    model: str,
) -> StreamingResponse:
    """Handle streaming Anthropic response.

    We reuse the CodeBuddy stream service which returns OpenAI SSE chunks,
    then convert each chunk to Anthropic SSE events on the fly.
    """
    converter = AnthropicStreamConverter(model)

    async def anthropic_stream():
        try:
            # The CodeBuddyStreamService.handle_stream_response returns a
            # StreamingResponse. We need to call the underlying stream_core
            # directly to get raw chunks and convert them.
            client = await _get_stream_client()
            async with client.stream(
                "POST", get_codebuddy_api_url(), json=payload, headers=headers
            ) as response:
                if response.status_code != 200:
                    error_bytes = await response.aread()
                    error_msg = error_bytes.decode("utf-8", errors="ignore")
                    error_event = _anthropic_error_sse(
                        "api_error",
                        f"CodeBuddy API error: {response.status_code} - {error_msg}",
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

                # Emit final events
                yield converter.finalize()

        except Exception as e:
            logger.error(f"Anthropic stream error: {e}", exc_info=True)
            yield _anthropic_error_sse("api_error", str(e))

    return StreamingResponse(
        anthropic_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _get_stream_client():
    """Get the shared HTTP client for streaming."""
    from .codebuddy_router import get_http_client
    return await get_http_client()


def _anthropic_error_sse(error_type: str, message: str) -> str:
    """Format an Anthropic-style error SSE event."""
    event_data = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }
    return f"event: error\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# GET /v1/models  — Anthropic models list
# ---------------------------------------------------------------------------

@router.get("/v1/models")
async def list_anthropic_models(request: Request):
    """List models in Anthropic format."""
    _ = _anthropic_auth(request)
    try:
        models = models_manager.get_available_models()
        response = convert_openai_models_to_anthropic(models)
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"获取Anthropic模型列表错误: {e}")
        error_body = anthropic_error_response(
            500, "api_error", "Failed to list models"
        )
        return JSONResponse(status_code=500, content=error_body)
