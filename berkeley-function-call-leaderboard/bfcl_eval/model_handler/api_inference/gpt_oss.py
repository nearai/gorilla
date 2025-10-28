import os
import json
from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from openai import OpenAI
import httpx
from bfcl_eval.model_handler.utils import convert_to_function_call, default_decode_execute_prompting





class GPTOSSInternalAPIHandler(OpenAICompletionsHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.client = OpenAI(
            api_key='secret123',
            base_url="http://154.57.34.78:8002/v1/",
            timeout=httpx.Timeout(timeout=30000.0, connect=8.0),
        )
    
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        Override to handle empty tool_calls properly
        """
        message = api_response.choices[0].message
        
        model_responses = []
        tool_call_ids = []
        
        # Explicit check instead of try/except
        if hasattr(message, 'tool_calls') and message.tool_calls and len(message.tool_calls) > 0:
            # Has tool calls
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in message.tool_calls
            ]
            tool_call_ids = [func_call.id for func_call in message.tool_calls]
        else:
            # No tool calls - use text content
            model_responses = message.content if hasattr(message, 'content') and message.content else ""
            tool_call_ids = []

        model_responses_message_for_chat_history = message

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }
    
    def decode_execute(self, result, has_tool_call_tag):
        """
        Override to handle both tool calls (list) and text responses (string)
        """
        # If result is a string (final text answer), this means NO function calls
        # Return empty list to signal we're done
        if isinstance(result, str):
            return []  # ← Changed from [result] to []
        
        # If result is a list (tool calls), use standard processing
        if self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result, has_tool_call_tag)