import re
import time
from typing import Any

import litellm
import openai
from openai import OpenAI

from typess import MessageList, SamplerBase, SamplerResponse

litellm.suppress_debug_info = True

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        base_url: str = None,
        completion_args: dict[str, Any] = {},
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI(base_url=base_url)
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.completion_args = completion_args
        if "hosted_vllm" in model:
            self.completion_args["api_key"] = "EMPTY"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=message_list,
                    **self.completion_args,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                sep_content = self.separate_thinking_from_response(content)
                return SamplerResponse(
                    response_text=sep_content["response"],
                    response_metadata={
                        "usage": response.usage,
                        "thinking": sep_content["thinking"],
                    },
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except litellm.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except ValueError as e:
                print("Value Error", e)
                return SamplerResponse(
                    response_text="No response (value error).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                print(
                    f"Unknown exception, returning error",
                    e,
                )
                return SamplerResponse(
                    response_text="No response (value error).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            # unknown error shall throw exception

    @staticmethod
    def separate_thinking_from_response(
        response: str,
        beginning_thinking_tag: str = "<think>",
        end_thinking_tag: str = "</think>",
    ) -> dict[str, str]:
        # regex for getting the content between the tags into thinking and the content after the end tag into response. Using only regex (no split) to avoid issues if the tags appear multiple times
        thinking_match = re.search(
            f"{re.escape(beginning_thinking_tag)}(.*?){re.escape(end_thinking_tag)}",
            response,
            re.DOTALL,
        )
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            actual_response = (
                response[thinking_match.end() :].strip()
                if response[thinking_match.end() :].strip()
                else ""
            )
        else:
            reponse_parts = response.split(end_thinking_tag)
            if len(reponse_parts) > 1:
                thinking = reponse_parts[0].replace(beginning_thinking_tag, "").strip()
                actual_response = reponse_parts[1].strip()
            else:
                thinking = ""
                actual_response = response.strip()
        return {"response": actual_response.strip(), "thinking": thinking}
