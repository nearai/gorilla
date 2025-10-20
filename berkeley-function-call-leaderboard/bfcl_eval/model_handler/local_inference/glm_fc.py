import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class GLMFCHandler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """
        Extract tool calls from GLM-4.5's output format
        """
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        """
        Convert tool calls to executable format
        """
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """
        Format prompt according to GLM-4.5's chat template
        Based on: https://huggingface.co/zai-org/GLM-4.5/blob/main/tokenizer_config.json
        """
        formatted_prompt = "[gMASK]<sop>"
        
        # Add tools section if functions provided
        if len(function) > 0:
            formatted_prompt += "<|system|>\n# Tools\n\n"
            formatted_prompt += "You may call one or more functions to assist with the user query.\n\n"
            formatted_prompt += "You are provided with function signatures within <tools></tools> XML tags:\n"
            formatted_prompt += "<tools>\n"
            for tool in function:
                formatted_prompt += json.dumps(tool, ensure_ascii=False) + "\n"
            formatted_prompt += "</tools>\n\n"
            formatted_prompt += "For each function call, output the function name and arguments within the following XML format:\n"
            formatted_prompt += "<tool_call>{function-name}\n"
            formatted_prompt += "<arg_key>{arg-key-1}</arg_key>\n"
            formatted_prompt += "<arg_value>{arg-value-1}</arg_value>\n"
            formatted_prompt += "<arg_key>{arg-key-2}</arg_key>\n"
            formatted_prompt += "<arg_value>{arg-value-2}</arg_value>\n"
            formatted_prompt += "...\n</tool_call>"

        # Find last user message index
        last_user_index = -1
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                last_user_index = idx

        # Format messages
        for idx, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            
            if isinstance(content, list):
                # Extract text from content array
                content = " ".join([
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ])
            
            if role == "user":
                formatted_prompt += f"<|user|>\n{content}"
                
            elif role == "assistant":
                formatted_prompt += "<|assistant|>"
                
                # Handle reasoning content
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]
                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = parts[-1].lstrip("\n")
                
                # Add thinking tags
                if idx > last_user_index and reasoning_content:
                    formatted_prompt += f"\n<think>{reasoning_content.strip()}</think>"
                else:
                    formatted_prompt += "\n<think></think>"
                
                # Add content
                if content.strip():
                    formatted_prompt += f"\n{content.strip()}"
                
                # Add tool calls
                if "tool_calls" in message and message["tool_calls"]:
                    for tc in message["tool_calls"]:
                        if "function" in tc:
                            tc = tc["function"]
                        
                        formatted_prompt += f"\n<tool_call>{tc['name']}\n"
                        
                        # Parse arguments if string
                        args = tc["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        
                        # Add arguments
                        for k, v in args.items():
                            formatted_prompt += f"<arg_key>{k}</arg_key>\n"
                            if not isinstance(v, str):
                                v = json.dumps(v, ensure_ascii=False)
                            formatted_prompt += f"<arg_value>{v}</arg_value>\n"
                        
                        formatted_prompt += "</tool_call>"
                        
            elif role == "tool":
                # Check if first tool message in sequence
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|observation|>"
                
                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"
                
            elif role == "system":
                formatted_prompt += f"<|system|>\n{content}"

        # Add generation prompt
        formatted_prompt += "<|assistant|>\n<think></think>"
        
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Parse model response, extracting tool calls or text content
        """
        model_response = api_response.choices[0].text
        
        # Extract tool calls
        extracted_tool_calls = self._extract_tool_calls(model_response)
        
        # Extract reasoning content
        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        # Build chat history message
        if len(extracted_tool_calls) > 0:
            # Has tool calls
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }
        else:
            # No tool calls - just text content
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }
        
        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        """
        Extract tool calls from GLM-4.5's XML format:
        <tool_call>{function-name}
        <arg_key>{key}</arg_key>
        <arg_value>{value}</arg_value>
        ...
        </tool_call>
        """
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)
        
        result = []
        for match in matches:
            try:
                # Split into lines
                lines = match.strip().split("\n")
                if not lines:
                    continue
                
                # First line is function name
                function_name = lines[0].strip()
                
                # Parse arguments
                arguments = {}
                i = 1
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Look for arg_key
                    if line.startswith("<arg_key>") and line.endswith("</arg_key>"):
                        key = line[9:-10]  # Extract between tags
                        
                        # Next line should be arg_value
                        if i + 1 < len(lines):
                            value_line = lines[i + 1].strip()
                            if value_line.startswith("<arg_value>") and value_line.endswith("</arg_value>"):
                                value = value_line[11:-12]  # Extract between tags
                                
                                # Try to parse as JSON if it looks like JSON
                                try:
                                    if value.startswith(("{", "[")) or value in ("true", "false", "null"):
                                        value = json.loads(value)
                                except:
                                    pass
                                
                                arguments[key] = value
                                i += 2
                                continue
                    
                    i += 1
                
                result.append({
                    "name": function_name,
                    "arguments": arguments
                })
            except Exception as e:
                print(f"Error parsing tool call: {e}")
                continue
        
        return result