"""
Anthropic Format Converter

Converts between OpenAI/CodeBuddy format and Anthropic Messages API format.
This enables codebuddy2api to expose an Anthropic-compatible endpoint
(/v1/messages) alongside the existing OpenAI-compatible (/v1/chat/completions).

Key differences handled:
- Anthropic separates system prompt from messages (top-level `system` field)
- Anthropic uses `content` blocks (text/tool_use/tool_result) instead of strings
- Anthropic uses `tool_use` / `tool_result` blocks vs OpenAI `tool_calls` / `tool` role
- Anthropic streaming uses event types: message_start, content_block_start,
  content_block_delta, content_block_stop, message_delta, message_stop
- Anthropic uses `max_tokens` (required) vs OpenAI's optional `max_tokens`
- Anthropic uses `stop_sequences` vs OpenAI's `stop`
- Tool call IDs: Anthropic uses `toolu_xxx`, OpenAI uses `call_xxx`
"""
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> OpenAI (CodeBuddy)
# ---------------------------------------------------------------------------

def convert_anthropic_request_to_openai(anthropic_body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic /v1/messages request body into an OpenAI
    /v1/chat/completions request body understood by the CodeBuddy backend.
    """
    openai_messages: List[Dict[str, Any]] = []

    # Anthropic puts the system prompt at top level (string or list of blocks)
    system_content = anthropic_body.get("system")
    if system_content:
        if isinstance(system_content, list):
            # list of content blocks, extract text
            parts = []
            for block in system_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            system_text = "\n".join(parts)
        else:
            system_text = str(system_content)
        if system_text:
            openai_messages.append({"role": "system", "content": system_text})

    # Convert each Anthropic message to OpenAI format
    for msg in anthropic_body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        converted = _convert_anthropic_message_to_openai(role, content)
        # converted may be a single message or a list of messages
        if isinstance(converted, list):
            openai_messages.extend(converted)
        else:
            openai_messages.append(converted)

    openai_body: Dict[str, Any] = {
        "model": anthropic_body.get("model", "auto-chat"),
        "messages": openai_messages,
        "stream": anthropic_body.get("stream", False),
    }

    # Parameter mapping
    if "max_tokens" in anthropic_body:
        openai_body["max_tokens"] = anthropic_body["max_tokens"]
    if "temperature" in anthropic_body:
        openai_body["temperature"] = anthropic_body["temperature"]
    if "top_p" in anthropic_body:
        openai_body["top_p"] = anthropic_body["top_p"]
    if "stop_sequences" in anthropic_body:
        openai_body["stop"] = anthropic_body["stop_sequences"]

    # Convert Anthropic tools -> OpenAI tools
    if anthropic_body.get("tools"):
        openai_body["tools"] = [
            _convert_anthropic_tool_to_openai(t) for t in anthropic_body["tools"]
        ]

    # tool_choice mapping
    tc = anthropic_body.get("tool_choice")
    if tc:
        if isinstance(tc, dict) and tc.get("type") == "auto":
            openai_body["tool_choice"] = "auto"
        elif isinstance(tc, dict) and tc.get("type") == "any":
            openai_body["tool_choice"] = "required"
        elif isinstance(tc, dict) and tc.get("type") == "tool":
            openai_body["tool_choice"] = {
                "type": "function",
                "function": {"name": tc.get("name", "")},
            }
        elif isinstance(tc, dict) and tc.get("type") == "none":
            openai_body["tool_choice"] = "none"

    return openai_body


def _convert_anthropic_message_to_openai(
    role: str, content: Any
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert a single Anthropic message to OpenAI message format.

    Returns a single message dict, or a list of messages when the Anthropic
    message contains tool_result blocks (which must be emitted as separate
    ``role: tool`` messages in OpenAI format, following the assistant message
    with tool_calls).
    """
    # Simple string content
    if isinstance(content, str):
        return {"role": role, "content": content}

    # content is a list of blocks
    if not isinstance(content, list):
        return {"role": role, "content": str(content)}

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            # Map Anthropic thinking -> OpenAI reasoning_content (assistant only)
            if role == "assistant":
                thinking_text = block.get("thinking", "")
                if thinking_text and thinking_text.strip():
                    reasoning_parts.append(thinking_text)
        elif block_type == "tool_use":
            tool_calls.append({
                "id": _convert_anthropic_tool_id_to_openai(block.get("id", "")),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(
                        block.get("input", {}), ensure_ascii=False
                    ),
                },
            })
        elif block_type == "tool_result":
            tool_use_id = block.get("tool_use_id", "")
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                parts = []
                for rb in result_content:
                    if isinstance(rb, dict) and rb.get("type") == "text":
                        parts.append(rb.get("text", ""))
                    elif isinstance(rb, str):
                        parts.append(rb)
                result_content = "\n".join(parts)
            tool_results.append({
                "role": "tool",
                "tool_call_id": _convert_anthropic_tool_id_to_openai(tool_use_id),
                "content": str(result_content),
            })

    # Build message(s)
    if role == "assistant":
        msg: Dict[str, Any] = {"role": "assistant"}
        text_content = "\n".join(text_parts)
        msg["content"] = text_content if text_content else None
        if reasoning_parts:
            msg["reasoning_content"] = "\n\n".join(reasoning_parts)
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg
    else:
        # user role - may contain text and/or tool results
        # In OpenAI format, tool results must be separate messages with role=tool
        messages: List[Dict[str, Any]] = []
        if text_parts:
            messages.append({"role": "user", "content": "\n".join(text_parts)})
        if tool_results:
            messages.extend(tool_results)
        if not messages:
            messages.append({"role": "user", "content": ""})
        return messages if len(messages) > 1 else messages[0]


def _convert_anthropic_tool_to_openai(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic tool definition to OpenAI tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _convert_anthropic_tool_id_to_openai(tool_id: str) -> str:
    """Convert toolu_xxx -> call_xxx for OpenAI compatibility."""
    if tool_id.startswith("toolu_"):
        return f"call_{tool_id[6:]}"
    return tool_id


def _convert_openai_tool_id_to_anthropic(tool_id: str) -> str:
    """Convert call_xxx -> toolu_xxx for Anthropic compatibility."""
    if tool_id.startswith("call_"):
        return f"toolu_{tool_id[5:]}"
    if tool_id.startswith("tooluse_"):
        return f"toolu_{tool_id[8:]}"
    return tool_id


# ---------------------------------------------------------------------------
# Response conversion: OpenAI -> Anthropic (non-stream)
# ---------------------------------------------------------------------------

def convert_openai_response_to_anthropic(
    openai_response: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """Convert an OpenAI chat completion response to Anthropic format."""
    choices = openai_response.get("choices", [])
    if not choices:
        return _error_anthropic_response("No choices in response", model)

    choice = choices[0]
    message = choice.get("message", {})
    content_blocks: List[Dict[str, Any]] = []

    # Reasoning content -> thinking block
    reasoning = message.get("reasoning_content")
    if reasoning:
        if isinstance(reasoning, list):
            for r in reasoning:
                if isinstance(r, dict):
                    text = r.get("text", "")
                else:
                    text = str(r)
                if text and text.strip():
                    content_blocks.append({"type": "thinking", "thinking": text})
        elif isinstance(reasoning, str) and reasoning.strip():
            content_blocks.append({"type": "thinking", "thinking": reasoning})

    # Text content
    text_content = message.get("content")
    if text_content:
        content_blocks.append({"type": "text", "text": text_content})

    # Tool calls
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_data = {}
        content_blocks.append({
            "type": "tool_use",
            "id": _convert_openai_tool_id_to_anthropic(tc.get("id", "")),
            "name": func.get("name", ""),
            "input": input_data,
        })

    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = _convert_openai_finish_to_anthropic(finish_reason)

    # If there were tool calls but finish_reason doesn't reflect it, override
    if tool_calls and finish_reason != "tool_calls":
        stop_reason = "tool_use"

    usage = openai_response.get("usage", {})

    return {
        "id": openai_response.get("id", str(uuid.uuid4())),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _convert_openai_finish_to_anthropic(finish: str) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(finish, "end_turn")


def _error_anthropic_response(message: str, model: str) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": "error",
        "model": model,
        "content": [{"type": "text", "text": f"Error: {message}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Response conversion: OpenAI SSE stream -> Anthropic SSE stream
# ---------------------------------------------------------------------------

class AnthropicStreamConverter:
    """Converts OpenAI SSE chunks to Anthropic SSE events.

    Anthropic streaming event sequence:
    1. event: message_start       (message metadata)
    2. event: content_block_start (per content block)
    3. event: content_block_delta (incremental text/tool args)
    4. event: content_block_stop  (per content block)
    5. event: message_delta       (stop_reason, usage)
    6. event: message_stop        (end)
    """

    def __init__(self, model: str):
        self.model = model
        self.message_id = str(uuid.uuid4())
        self.block_index = 0
        self.current_block_type: Optional[str] = None  # "text", "tool_use", or "thinking"
        self.current_tool_id: Optional[str] = None
        self.current_tool_name: Optional[str] = None
        self.current_tool_args = ""
        self.started = False
        self.output_tokens = 0
        self.finish_reason: Optional[str] = None
        self.saw_tool_call = False

    def _format_event(self, event_type: str, data: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _start_message_if_needed(self) -> Optional[str]:
        if self.started:
            return None
        self.started = True
        return self._format_event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        })

    def _start_text_block(self) -> str:
        self.current_block_type = "text"
        events = self._start_message_if_needed()
        event = self._format_event("content_block_start", {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {"type": "text", "text": ""},
        })
        return (events or "") + event

    def _start_thinking_block(self) -> str:
        self.current_block_type = "thinking"
        events = self._start_message_if_needed()
        event = self._format_event("content_block_start", {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {"type": "thinking", "thinking": ""},
        })
        return (events or "") + event

    def _start_tool_block(self, tool_id: str, tool_name: str) -> str:
        self.current_block_type = "tool_use"
        self.current_tool_id = tool_id
        self.current_tool_name = tool_name
        self.current_tool_args = ""
        events = self._start_message_if_needed()
        # If we were in a text block, close it first
        event_parts = []
        if events:
            event_parts.append(events)
        event_parts.append(self._format_event("content_block_start", {
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": {},
            },
        }))
        return "".join(event_parts)

    def _stop_current_block(self) -> Optional[str]:
        if self.current_block_type is None:
            return None
        event = self._format_event("content_block_stop", {
            "type": "content_block_stop",
            "index": self.block_index,
        })
        self.current_block_type = None
        self.block_index += 1
        return event

    def convert_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Convert a single OpenAI SSE chunk to Anthropic SSE events.

        Returns a string of zero or more SSE event blocks.
        """
        parts: List[str] = []

        choices = chunk_data.get("choices", [])
        if not choices:
            return ""

        choice = choices[0]
        delta = choice.get("delta", {})

        # Reasoning content -> thinking_delta
        reasoning = delta.get("reasoning_content")
        if reasoning:
            reasoning_texts = _collect_reasoning_texts(reasoning)
            for reasoning_text in reasoning_texts:
                if not reasoning_text:
                    continue
                if self.current_block_type != "thinking":
                    stop_event = self._stop_current_block()
                    if stop_event:
                        parts.append(stop_event)
                    parts.append(self._start_thinking_block())
                parts.append(self._format_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning_text},
                }))

        # Text content
        content = delta.get("content")
        if content:
            if self.current_block_type != "text":
                # Close any open tool block first
                stop_event = self._stop_current_block()
                if stop_event:
                    parts.append(stop_event)
                parts.append(self._start_text_block())
            parts.append(self._format_event("content_block_delta", {
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": {"type": "text_delta", "text": content},
            }))
            self.output_tokens += max(1, len(content) // 4)

        # Tool calls
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            tc_id = tc.get("id")
            tc_name = func.get("name")
            tc_args = func.get("arguments", "")

            # New tool call (has id and/or name)
            if tc_id or tc_name:
                # Close previous block
                stop_event = self._stop_current_block()
                if stop_event:
                    parts.append(stop_event)

                tool_id_anthropic = _convert_openai_tool_id_to_anthropic(
                    tc_id or f"call_{uuid.uuid4().hex[:12]}"
                )
                parts.append(self._start_tool_block(
                    tool_id_anthropic,
                    tc_name or "",
                ))
                self.saw_tool_call = True
                if tc_args:
                    parts.append(self._format_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tc_args,
                        },
                    }))
                    self.current_tool_args += tc_args
            elif tc_args and self.current_block_type == "tool_use":
                # Incremental tool arguments
                parts.append(self._format_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": tc_args,
                    },
                }))
                self.current_tool_args += tc_args

        # Finish reason
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self.finish_reason = finish_reason

        # Usage info
        usage = chunk_data.get("usage")
        if usage:
            self.output_tokens = usage.get("completion_tokens", self.output_tokens)

        return "".join(parts)

    def finalize(self) -> str:
        """Generate closing events for the stream."""
        parts: List[str] = []

        # Close any open block
        stop_event = self._stop_current_block()
        if stop_event:
            parts.append(stop_event)

        # message_delta with stop_reason
        effective_finish = self.finish_reason or "stop"
        if self.saw_tool_call:
            effective_finish = "tool_calls"
        stop_reason = _convert_openai_finish_to_anthropic(effective_finish)
        parts.append(self._format_event("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
            },
            "usage": {
                "input_tokens": 0,
                "output_tokens": self.output_tokens,
            },
        }))

        # message_stop
        parts.append(self._format_event("message_stop", {
            "type": "message_stop",
        }))

        return "".join(parts)


# ---------------------------------------------------------------------------
# Anthropic error response helper
# ---------------------------------------------------------------------------

def anthropic_error_response(
    status_code: int,
    error_type: str,
    message: str,
) -> Dict[str, Any]:
    """Build an Anthropic-style error response body."""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def _collect_reasoning_texts(reasoning: Any) -> List[str]:
    """Extract text strings from reasoning_content which may be a string,
    a single object, or a list of objects."""
    texts: List[str] = []
    if not reasoning:
        return texts

    if isinstance(reasoning, str):
        if reasoning.strip():
            texts.append(reasoning)
        return texts

    if isinstance(reasoning, list):
        for item in reasoning:
            texts.extend(_collect_reasoning_texts(item))
        return texts

    if isinstance(reasoning, dict):
        text = reasoning.get("text")
        if text and isinstance(text, str) and text.strip():
            texts.append(text)
        return texts

    return texts


# ---------------------------------------------------------------------------
# Anthropic models list response
# ---------------------------------------------------------------------------

def convert_openai_models_to_anthropic(
    openai_models: List[str],
) -> Dict[str, Any]:
    """Convert an OpenAI model list to Anthropic /v1/models format."""
    now = int(time.time())
    return {
        "data": [
            {
                "id": model,
                "type": "model",
                "display_name": model,
                "created_at": now,
            }
            for model in openai_models
        ],
        "first_id": openai_models[0] if openai_models else None,
        "has_more": False,
        "last_id": openai_models[-1] if openai_models else None,
    }
