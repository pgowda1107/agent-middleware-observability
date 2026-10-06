# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

from pydantic import ConfigDict

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
)


class RelayHandoffResourcesServerConfig(BaseResourcesServerConfig):
    pass


class RelayHandoffVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class RelayHandoffVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class RelayHandoffResourcesServer(SimpleResourcesServer):
    config: RelayHandoffResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/research_agent")(self.research_agent)
        app.post("/writer_agent")(self.writer_agent)

        return app

    async def research_agent(self, body: dict[str, Any]) -> dict[str, str]:
        task = str(body.get("task", ""))
        return {"output": f"research complete: {task}"}

    async def writer_agent(self, body: dict[str, Any]) -> dict[str, str]:
        task = str(body.get("task", ""))
        return {"output": f"writer complete: {task}"}

    async def verify(self, body: RelayHandoffVerifyRequest) -> RelayHandoffVerifyResponse:
        task_data = body.model_extra.get("task_data", {}) if body.model_extra else {}
        expected = [
            item["from"]
            for item in task_data.get("expected_handoffs", [])
            if isinstance(item, dict) and item.get("from")
        ]
        actual = [item.name for item in body.response.output if item.type == "function_call"]
        missing = [name for name in expected if name not in actual]
        ordered = _is_ordered_subset(expected, actual)
        reward = 1.0 if expected and not missing and ordered else 0.0
        payload = body.model_dump()
        payload.update(
            {
                "reward": reward,
                "expected_handoff_tools": expected,
                "actual_handoff_tools": actual,
                "missing_handoff_tools": missing,
                "handoff_order_ok": ordered,
            }
        )
        return RelayHandoffVerifyResponse(**payload)


def _is_ordered_subset(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return False
    position = 0
    for item in actual:
        if position < len(expected) and item == expected[position]:
            position += 1
    return position == len(expected)


if __name__ == "__main__":
    RelayHandoffResourcesServer.run_webserver()
