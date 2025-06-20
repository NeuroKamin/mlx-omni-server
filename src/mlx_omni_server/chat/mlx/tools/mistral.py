import json
import uuid
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ...schema import ChatMessage, FunctionCall, Role, ToolCall
from .chat_tokenizer import ChatTokenizer


class MistralChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = "[TOOL_CALLS]"
        self.end_tool_calls = ""

    def decode_stream(self, text: str, delta_text: str) -> Optional[List[ToolCall]]:
        pass

    def decode(self, text: str) -> Optional[ChatMessage]:
        """Parse tool calls from model output.

        The model outputs function calls in the format:
        [TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}},
                     {"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}]

        Args:
            text: The model output text containing tool calls

        Returns:
            ChatMessage: A message containing the parsed tool calls
        """
        # Look for JSON patterns in the text
        tool_calls = []

        if text.startswith(self.start_tool_calls):
            try:
                # Extract the JSON array from between square brackets after [TOOL_CALLS]
                json_str = text[len(self.start_tool_calls) :].strip()
                if json_str.startswith("[") and json_str.endswith("]"):
                    json_str = json_str.strip("[]").strip()
                    # Try to parse as array first
                    try:
                        tool_data = json.loads(f"[{json_str}]")
                    except json.JSONDecodeError:
                        # If array parsing fails, try single object
                        tool_data = json.loads(json_str)

                    # Handle both single object and array of objects
                    if isinstance(tool_data, dict):
                        tool_data = [tool_data]
                    elif not isinstance(tool_data, list):
                        raise ValueError(
                            "Invalid tool call format: expected dict or list"
                        )

                    for call in tool_data:
                        if not isinstance(call, dict) or "name" not in call:
                            continue

                        # Get arguments and ensure they're a JSON string
                        args = call.get("arguments", call.get("parameters", {}))
                        
                        # Handle null arguments - convert to empty JSON object
                        if args is None or (isinstance(args, str) and args == "null"):
                            arguments = "{}"
                        elif isinstance(args, str):
                            # Validate it's proper JSON, if not make it empty object
                            try:
                                json.loads(args)  # Validate JSON
                                arguments = args
                            except json.JSONDecodeError:
                                arguments = "{}"
                        else:
                            # Convert dict to JSON string
                            arguments = json.dumps(args)

                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                function=FunctionCall(
                                    name=call["name"],
                                    arguments=arguments,
                                ),
                            )
                        )
                else:
                    # Invalid format, return original text
                    return ChatMessage(
                        role=Role.ASSISTANT,
                        content=text,
                        tool_calls=None,
                    )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing tool call: {e}")
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=text,
                    tool_calls=None,
                )

        return ChatMessage(
            role=Role.ASSISTANT,
            content=None if tool_calls else text,
            tool_calls=tool_calls if tool_calls else None,
        )
