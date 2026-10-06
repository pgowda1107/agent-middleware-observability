from __future__ import annotations

import json
from time import time
from uuid import uuid4

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


class ScriptedHandoffModelConfig(BaseResponsesAPIModelConfig):
    model: str = "scripted-handoff-model"


class ScriptedHandoffModel(SimpleResponsesAPIModel):
    config: ScriptedHandoffModelConfig

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        outputs = list(body.input) if isinstance(body.input, list) else []
        tool_output_count = sum(1 for item in outputs if getattr(item, "type", None) == "function_call_output")

        if tool_output_count == 0:
            output = [
                {
                    "type": "function_call",
                    "call_id": "call_research_agent",
                    "name": "research_agent",
                    "arguments": json.dumps({"task": _last_user_text(body)}),
                }
            ]
        elif tool_output_count == 1:
            output = [
                {
                    "type": "function_call",
                    "call_id": "call_writer_agent",
                    "name": "writer_agent",
                    "arguments": json.dumps({"task": "Write a concise summary from the research result."}),
                }
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": f"msg_{uuid4().hex}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The Relay handoff completed through research_agent then writer_agent.",
                            "annotations": [],
                        }
                    ],
                }
            ]

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=output,
            parallel_tool_calls=False,
            tool_choice=body.tool_choice if body.tool_choice is not None else "auto",
            tools=body.tools,
        )

    async def chat_completions(
        self,
        body: NeMoGymChatCompletionCreateParamsNonStreaming = Body(),
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError("scripted_handoff_model only implements /v1/responses")


def _last_user_text(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    if isinstance(body.input, str):
        return body.input
    for item in reversed(body.input):
        if getattr(item, "role", None) == "user":
            return str(getattr(item, "content", ""))
    return "Research the Relay handoff, then write a concise summary."


if __name__ == "__main__":
    ScriptedHandoffModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ScriptedHandoffModel.run_webserver()  # noqa: F401
