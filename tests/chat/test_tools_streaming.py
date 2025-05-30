import logging
import json
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


def get_weather_tools():
    """Return sample weather tools for testing"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]


class TestToolsStreaming:
    """Test class to verify tool calls work the same way in streaming and non-streaming modes"""

    def test_tool_calls_comparison(self, openai_client):
        """Test that tool calls produce the same structure in both streaming and non-streaming modes"""
        
        model = "mlx-community/gemma-3-1b-it-4bit-DWQ"
        tools = get_weather_tools()
        messages = [
            {"role": "user", "content": "What's the weather like in Boston?"}
        ]
        
        # Test non-streaming mode
        response_normal = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False
        )
        
        logger.info(f"Non-streaming response: {response_normal}")
        
        # Test streaming mode
        stream = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True
        )
        
        # Collect streaming chunks
        streaming_chunks = []
        streaming_content = ""
        streaming_tool_calls = None
        
        for chunk in stream:
            streaming_chunks.append(chunk)
            logger.info(f"Streaming chunk: {chunk}")
            
            choice = chunk.choices[0]
            if choice.delta.content:
                streaming_content += choice.delta.content
            
            # Check for tool calls in delta
            if choice.delta.tool_calls:
                streaming_tool_calls = choice.delta.tool_calls
        
        # Compare results
        normal_choice = response_normal.choices[0]
        
        logger.info(f"Normal finish_reason: {normal_choice.finish_reason}")
        logger.info(f"Normal tool_calls: {normal_choice.message.tool_calls}")
        logger.info(f"Streaming tool_calls: {streaming_tool_calls}")
        
        # Validate that both modes produce similar results
        if normal_choice.message.tool_calls:
            # If non-streaming detected tool calls, streaming should too
            assert streaming_tool_calls is not None, "Streaming mode should detect tool calls when non-streaming does"
            assert normal_choice.finish_reason == "tool_calls", "Non-streaming should have tool_calls finish_reason"
            
            # Find the chunk with tool_calls finish_reason in streaming
            tool_call_chunks = [chunk for chunk in streaming_chunks 
                               if chunk.choices[0].finish_reason == "tool_calls"]
            assert len(tool_call_chunks) > 0, "Streaming should have chunk with tool_calls finish_reason"
            
            # Compare tool call structure
            normal_tool_call = normal_choice.message.tool_calls[0]
            streaming_tool_call = streaming_tool_calls[0]
            
            assert normal_tool_call.function.name == streaming_tool_call.function.name, \
                "Tool call names should match between streaming and non-streaming"
            
            # Compare arguments (parse JSON to handle formatting differences)
            normal_args = json.loads(normal_tool_call.function.arguments)
            streaming_args = json.loads(streaming_tool_call.function.arguments)
            assert normal_args == streaming_args, \
                "Tool call arguments should match between streaming and non-streaming"
        
        else:
            # If non-streaming didn't detect tool calls, streaming shouldn't either
            # (or it's a case where the model didn't use tools)
            logger.info("No tool calls detected in non-streaming mode")

    def test_tool_calls_required_mode(self, openai_client):
        """Test tool calls in required mode for both streaming and non-streaming"""
        
        model = "mlx-community/gemma-3-1b-it-4bit-DWQ"
        tools = get_weather_tools()
        messages = [
            {"role": "user", "content": "Get weather for New York"}
        ]
        
        # Test non-streaming with required tool choice
        response_normal = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            stream=False
        )
        
        # Test streaming with required tool choice
        stream = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            stream=True
        )
        
        # Collect streaming results
        streaming_chunks = list(stream)
        
        # Both should produce tool calls when required
        normal_choice = response_normal.choices[0]
        
        logger.info(f"Required mode - Normal response: {response_normal}")
        logger.info(f"Required mode - Streaming chunks: {len(streaming_chunks)}")
        
        assert normal_choice.message.tool_calls is not None, \
            "Non-streaming should produce tool calls when required"
        assert normal_choice.finish_reason == "tool_calls", \
            "Non-streaming should have tool_calls finish_reason when required"
        
        # Check streaming results
        tool_call_chunks = [chunk for chunk in streaming_chunks 
                           if chunk.choices[0].delta.tool_calls is not None]
        assert len(tool_call_chunks) > 0, \
            "Streaming should produce tool calls when required" 