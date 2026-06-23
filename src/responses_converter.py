"""
OpenAI Responses API Format Converter

Converts between OpenAI Chat Completions format and OpenAI Responses API format.
This enables codebuddy2api to expose the /v1/responses endpoint (used by
Codex CLI and other tools that consume the Responses API).

Key differences between Responses API and Chat Completions:
- Responses uses `input` (array of items) instead of `messages`
- Responses uses `instructions` (top-level string) instead of system message
- Responses uses `max_output_tokens` instead of `max_tokens`
- Responses item types: message, function_call, function_call_output,
  custom_tool_call, custom_tool_call_output
- Responses streaming events: response.created, response.in_progress,
  response.output_item.added, response.content_part.added,
  response.output_text.delta, response.content_part.done,
  response.output_item.done, response.completed

Reference: CLIProxyAPI internal/translator/openai/openai/responses/
"""
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request conversion: Responses -> Chat Completions
# ---------------------------------------------------------------------------

def convert_responses_request_to_chat_completions(
    responses_body: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert an OpenAI Responses API request to Chat Completions format.

    This mirrors CLIProxyAPI's ConvertOpenAIResponsesRequestToOpenAIChatCompletions.
    """
    openai_messages: List[Dict[str, Any]] = []

    # instructions -> system message
    instructions = responses_body.get("instructions")
    if instructions:
        openai_messages.append({"role": "system", "content": instructions})

    # Convert input array to messages
    input_data = responses_body.get("input")

    if isinstance(input_data, str):
        # Simple string input
        openai_messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        pending_tool_calls: List[Dict[str, Any]] = []
        pending_tool_call_ids: List[str] = []

        def flush_pending_tool_calls():
            nonlocal pending_tool_calls, pending_tool_call_ids
            if not pending_tool_calls:
                return
            openai_messages.append({
                "role": "assistant",
                "tool_calls": pending_tool_calls,
            })
            pending_tool_calls = []
            pending_tool_call_ids = []

        for item in input_data:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")
            if not item_type and item.get("role"):
                item_type = "message"

            is_tool_call = item_type in (
                "function_call", "custom_tool_call", "tool_search_call"
            )
            is_tool_output = item_type in (
                "function_call_output", "custom_tool_call_output", "tool_search_output"
            )

            if not is_tool_call:
                flush_pending_tool_calls()

            if item_type in ("message", ""):
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                content = item.get("content")
                msg: Dict[str, Any] = {"role": role}

                if isinstance(content, str):
                    msg["content"] = content
                elif isinstance(content, list):
                    text_parts = []
                    for content_item in content:
                        if not isinstance(content_item, dict):
                            continue
                        ct = content_item.get("type", "input_text")
                        if ct in ("input_text", "output_text"):
                            text_parts.append(content_item.get("text", ""))
                        elif ct == "input_image":
                            # Simplified: pass as text description
                            pass
                    msg["content"] = "\n".join(text_parts) if text_parts else ""
                else:
                    msg["content"] = str(content) if content else ""

                openai_messages.append(msg)

            elif is_tool_call:
                # Buffer tool calls and emit as a single assistant message
                call_id = item.get("call_id", item.get("id", ""))
                name = item.get("name", "")
                if item_type == "custom_tool_call":
                    args = json.dumps(
                        {"input": item.get("input", "")}, ensure_ascii=False
                    )
                elif item_type == "tool_search_call":
                    args = item.get("arguments", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    name = "tool_search"
                else:
                    args = item.get("arguments", "{}")
                    if not args:
                        args = "{}"

                tool_call = {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
                pending_tool_calls.append(tool_call)
                if call_id:
                    pending_tool_call_ids.append(call_id)

            elif is_tool_output:
                call_id = item.get("call_id", "")
                output = item.get("output", "")
                if isinstance(output, (dict, list)):
                    output = json.dumps(output, ensure_ascii=False)
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(output),
                })

        flush_pending_tool_calls()

    openai_body: Dict[str, Any] = {
        "model": responses_body.get("model", "auto-chat"),
        "messages": openai_messages,
        "stream": responses_body.get("stream", False),
    }

    if stream := responses_body.get("stream"):
        openai_body["stream_options"] = {"include_usage": True}

    # max_output_tokens -> max_tokens
    if "max_output_tokens" in responses_body:
        openai_body["max_tokens"] = responses_body["max_output_tokens"]

    # temperature
    if "temperature" in responses_body:
        openai_body["temperature"] = responses_body["temperature"]
    if "top_p" in responses_body:
        openai_body["top_p"] = responses_body["top_p"]

    # parallel_tool_calls
    if "parallel_tool_calls" in responses_body:
        openai_body["parallel_tool_calls"] = responses_body["parallel_tool_calls"]

    # reasoning.effort -> reasoning_effort
    reasoning = responses_body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        openai_body["reasoning_effort"] = reasoning["effort"].lower()

    # tool_choice passthrough
    if "tool_choice" in responses_body:
        openai_body["tool_choice"] = responses_body["tool_choice"]

    # Convert tools: responses format -> chat completions format
    if responses_body.get("tools"):
        openai_body["tools"] = [
            _convert_responses_tool_to_chat_tool(t)
            for t in responses_body["tools"]
            if isinstance(t, dict)
        ]

    return openai_body


def _convert_responses_tool_to_chat_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Responses API tool definition to Chat Completions format."""
    tool_type = tool.get("type", "function")
    name = tool.get("name", "")

    if tool_type == "function":
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }
    elif tool_type == "custom":
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                },
            },
        }
    else:
        # Default to function
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }


# ---------------------------------------------------------------------------
# Response conversion: Chat Completions -> Responses (non-stream)
# ---------------------------------------------------------------------------

def convert_chat_completion_to_responses(
    openai_response: Dict[str, Any],
    model: str,
    original_request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a Chat Completions response to Responses API format (non-stream)."""
    response_id = openai_response.get("id", f"resp_{uuid.uuid4().hex[:24]}")
    created = openai_response.get("created", int(time.time()))

    output_items: List[Dict[str, Any]] = []

    choices = openai_response.get("choices", [])
    if choices:
        choice = choices[0]
        message = choice.get("message", {})

        # Text content -> message output item
        text_content = message.get("content")
        if text_content:
            output_items.append({
                "id": f"msg_{response_id}_0",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": text_content,
                    "annotations": [],
                }],
            })

        # Tool calls -> function_call items
        tool_calls = message.get("tool_calls", [])
        for i, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            call_id = tc.get("id", f"call_{i}")
            try:
                args = func.get("arguments", "{}")
                # Validate JSON
                json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = "{}"

            output_items.append({
                "id": f"fc_{call_id}",
                "type": "function_call",
                "status": "completed",
                "name": func.get("name", ""),
                "call_id": call_id,
                "arguments": args,
            })

    # Usage
    usage = openai_response.get("usage", {})
    responses_usage: Dict[str, Any] = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
    }

    result: Dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "background": False,
        "error": None,
        "output": output_items,
        "usage": responses_usage,
        "model": model,
    }

    # Inject original request fields if available
    if original_request:
        if v := original_request.get("instructions"):
            result["instructions"] = v
        if v := original_request.get("max_output_tokens"):
            result["max_output_tokens"] = v
        if v := original_request.get("temperature"):
            result["temperature"] = v
        if v := original_request.get("top_p"):
            result["top_p"] = v
        if v := original_request.get("tool_choice"):
            result["tool_choice"] = v
        if v := original_request.get("tools"):
            result["tools"] = v
        if v := original_request.get("reasoning"):
            result["reasoning"] = v
        if v := original_request.get("parallel_tool_calls"):
            result["parallel_tool_calls"] = v

    return result


# ---------------------------------------------------------------------------
# Response conversion: Chat Completions SSE stream -> Responses SSE stream
# ---------------------------------------------------------------------------

class ResponsesStreamConverter:
    """Converts OpenAI Chat Completions SSE chunks to Responses API SSE events.

    Responses streaming event sequence:
    1. response.created          (initial response object)
    2. response.in_progress      (status update)
    3. response.output_item.added        (per output item)
    4. response.content_part.added       (per content part in a message item)
    5. response.output_text.delta        (incremental text)
    6. response.content_part.done        (content part finished)
    7. response.output_item.done         (output item finished)
    8. response.completed        (final response with all output)
    """

    def __init__(self, model: str, original_request: Optional[Dict[str, Any]] = None):
        self.model = model
        self.original_request = original_request or {}
        self.response_id = ""
        self.created = 0
        self.started = False
        self.completed = False

        # Sequence number counter
        self.seq = 0

        # Output tracking
        self.next_output_index = 0
        self.current_msg_index = -1
        self.current_msg_text = []  # type: List[str]
        self.msg_started = False

        # Tool call tracking
        self.tool_calls: Dict[str, Dict[str, Any]] = {}  # key: tool_index
        self.tool_order: List[str] = []

        # Usage
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.usage_seen = False

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _format_event(self, event_type: str, data: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _start_if_needed(self, chunk_data: Dict[str, Any]) -> str:
        if self.started:
            return ""
        self.started = True

        if not self.response_id:
            self.response_id = chunk_data.get("id", f"resp_{uuid.uuid4().hex[:24]}")
        if not self.created:
            self.created = chunk_data.get("created", int(time.time()))

        parts: List[str] = []

        # response.created
        parts.append(self._format_event("response.created", {
            "type": "response.created",
            "sequence_number": self._next_seq(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created,
                "status": "in_progress",
                "background": False,
                "error": None,
                "output": [],
                "model": self.model,
            },
        }))

        # response.in_progress
        parts.append(self._format_event("response.in_progress", {
            "type": "response.in_progress",
            "sequence_number": self._next_seq(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created,
                "status": "in_progress",
            },
        }))

        return "".join(parts)

    def _start_message_item(self) -> str:
        self.current_msg_index = self.next_output_index
        self.next_output_index += 1
        self.msg_started = True
        self.current_msg_text = []

        msg_id = f"msg_{self.response_id}_{self.current_msg_index}"

        parts: List[str] = []

        # response.output_item.added
        parts.append(self._format_event("response.output_item.added", {
            "type": "response.output_item.added",
            "sequence_number": self._next_seq(),
            "output_index": self.current_msg_index,
            "item": {
                "id": msg_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }))

        # response.content_part.added
        parts.append(self._format_event("response.content_part.added", {
            "type": "response.content_part.added",
            "sequence_number": self._next_seq(),
            "item_id": msg_id,
            "output_index": self.current_msg_index,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
            },
        }))

        return "".join(parts)

    def _finish_message_item(self) -> str:
        if not self.msg_started:
            return ""

        self.msg_started = False
        msg_id = f"msg_{self.response_id}_{self.current_msg_index}"
        full_text = "".join(self.current_msg_text)

        parts: List[str] = []

        # response.content_part.done
        parts.append(self._format_event("response.content_part.done", {
            "type": "response.content_part.done",
            "sequence_number": self._next_seq(),
            "item_id": msg_id,
            "output_index": self.current_msg_index,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": full_text,
                "annotations": [],
            },
        }))

        # response.output_item.done
        parts.append(self._format_event("response.output_item.done", {
            "type": "response.output_item.done",
            "sequence_number": self._next_seq(),
            "output_index": self.current_msg_index,
            "item": {
                "id": msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": full_text,
                    "annotations": [],
                }],
            },
        }))

        return "".join(parts)

    def _start_tool_call(self, tool_key: str, call_id: str, name: str) -> str:
        if tool_key in self.tool_calls:
            return ""

        output_index = self.next_output_index
        self.next_output_index += 1

        self.tool_calls[tool_key] = {
            "output_index": output_index,
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "started": True,
        }
        self.tool_order.append(tool_key)

        item_id = f"fc_{call_id}"

        return self._format_event("response.output_item.added", {
            "type": "response.output_item.added",
            "sequence_number": self._next_seq(),
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "name": name,
                "call_id": call_id,
                "arguments": "",
            },
        })

    def _finish_tool_calls(self) -> str:
        parts: List[str] = []
        for tool_key in self.tool_order:
            tc = self.tool_calls[tool_key]
            if not tc.get("started"):
                continue
            tc["started"] = False

            item_id = f"fc_{tc['call_id']}"

            parts.append(self._format_event("response.output_item.done", {
                "type": "response.output_item.done",
                "sequence_number": self._next_seq(),
                "output_index": tc["output_index"],
                "item": {
                    "id": item_id,
                    "type": "function_call",
                    "status": "completed",
                    "name": tc["name"],
                    "call_id": tc["call_id"],
                    "arguments": tc["arguments"] or "{}",
                },
            }))
        return "".join(parts)

    def convert_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Convert a single Chat Completions SSE chunk to Responses SSE events."""
        parts: List[str] = []

        parts.append(self._start_if_needed(chunk_data))

        choices = chunk_data.get("choices", [])
        if not choices:
            # May contain usage info only
            usage = chunk_data.get("usage")
            if usage:
                self._process_usage(usage)
            return "".join(parts)

        choice = choices[0]
        delta = choice.get("delta", {})

        # Text content
        content = delta.get("content")
        if content:
            if not self.msg_started:
                parts.append(self._start_message_item())
            self.current_msg_text.append(content)
            msg_id = f"msg_{self.response_id}_{self.current_msg_index}"
            parts.append(self._format_event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "sequence_number": self._next_seq(),
                "item_id": msg_id,
                "output_index": self.current_msg_index,
                "content_index": 0,
                "delta": content,
            }))

        # Tool calls
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            tc_index = str(tc.get("index", 0))
            func = tc.get("function", {})
            tc_id = tc.get("id")
            tc_name = func.get("name")
            tc_args = func.get("arguments", "")

            # If we have an active message, close it first
            if self.msg_started:
                parts.append(self._finish_message_item())

            if tc_index not in self.tool_calls:
                call_id = tc_id or f"call_{uuid.uuid4().hex[:12]}"
                name = tc_name or ""
                parts.append(self._start_tool_call(tc_index, call_id, name))
            elif tc_name and not self.tool_calls[tc_index]["name"]:
                self.tool_calls[tc_index]["name"] = tc_name

            if tc_args:
                self.tool_calls[tc_index]["arguments"] += tc_args
                parts.append(self._format_event("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": self._next_seq(),
                    "item_id": f"fc_{self.tool_calls[tc_index]['call_id']}",
                    "output_index": self.tool_calls[tc_index]["output_index"],
                    "delta": tc_args,
                }))

        # Finish reason - close any open items
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            if self.msg_started:
                parts.append(self._finish_message_item())
            parts.append(self._finish_tool_calls())

        # Usage
        usage = chunk_data.get("usage")
        if usage:
            self._process_usage(usage)

        return "".join(parts)

    def _process_usage(self, usage: Dict[str, Any]):
        self.prompt_tokens = usage.get("prompt_tokens", 0)
        self.completion_tokens = usage.get("completion_tokens", 0)
        self.total_tokens = usage.get(
            "total_tokens", self.prompt_tokens + self.completion_tokens
        )
        self.usage_seen = True

    def finalize(self) -> str:
        """Generate the response.completed event."""
        if self.completed:
            return ""
        self.completed = True

        # Close any open items
        parts: List[str] = []
        if self.msg_started:
            parts.append(self._finish_message_item())
        parts.append(self._finish_tool_calls())

        # Build output array for completed event
        output_items: List[Dict[str, Any]] = []

        # Message item
        if self.current_msg_index >= 0 and self.current_msg_text:
            full_text = "".join(self.current_msg_text)
            output_items.append({
                "id": f"msg_{self.response_id}_{self.current_msg_index}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": full_text,
                    "annotations": [],
                }],
            })

        # Tool call items
        for tool_key in self.tool_order:
            tc = self.tool_calls[tool_key]
            output_items.append({
                "id": f"fc_{tc['call_id']}",
                "type": "function_call",
                "status": "completed",
                "name": tc["name"],
                "call_id": tc["call_id"],
                "arguments": tc["arguments"] or "{}",
            })

        # Build response object
        response_obj: Dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created,
            "status": "completed",
            "background": False,
            "error": None,
            "output": output_items,
            "model": self.model,
        }

        # Inject original request fields
        if v := self.original_request.get("instructions"):
            response_obj["instructions"] = v
        if v := self.original_request.get("max_output_tokens"):
            response_obj["max_output_tokens"] = v
        if v := self.original_request.get("temperature"):
            response_obj["temperature"] = v
        if v := self.original_request.get("top_p"):
            response_obj["top_p"] = v
        if v := self.original_request.get("tool_choice"):
            response_obj["tool_choice"] = v
        if v := self.original_request.get("tools"):
            response_obj["tools"] = v
        if v := self.original_request.get("reasoning"):
            response_obj["reasoning"] = v

        # Usage
        if self.usage_seen:
            response_obj["usage"] = {
                "input_tokens": self.prompt_tokens,
                "output_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }
        else:
            response_obj["usage"] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        parts.append(self._format_event("response.completed", {
            "type": "response.completed",
            "sequence_number": self._next_seq(),
            "response": response_obj,
        }))

        return "".join(parts)


# ---------------------------------------------------------------------------
# Error response helper
# ---------------------------------------------------------------------------

def responses_error_response(
    status_code: int,
    message: str,
) -> Dict[str, Any]:
    """Build a Responses API-style error response body."""
    return {
        "type": "error",
        "error": {
            "code": str(status_code),
            "message": message,
        },
    }
