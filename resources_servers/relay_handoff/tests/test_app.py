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
import pytest

from nemo_gym.server_utils import ServerClient

from resources_servers.relay_handoff.app import RelayHandoffResourcesServer, RelayHandoffResourcesServerConfig

from unittest.mock import MagicMock


class TestApp:
    def test_sanity(self) -> None:
        config = RelayHandoffResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        RelayHandoffResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

    @pytest.mark.asyncio
    async def test_verify_scores_expected_handoff_tools(self) -> None:
        config = RelayHandoffResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = RelayHandoffResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        request = {
            "responses_create_params": {
                "input": "Research the Relay handoff, then write a concise summary.",
                "parallel_tool_calls": False,
                "tools": [],
            },
            "response": {
                "id": "resp_1",
                "created_at": 0,
                "model": "test-model",
                "object": "response",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "research_agent",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "writer_agent",
                        "arguments": "{}",
                    },
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            "task_data": {
                "expected_handoffs": [
                    {"from": "research_agent", "to": "subagent:research"},
                    {"from": "writer_agent", "to": "subagent:writer"},
                ]
            },
        }

        response = await server.verify(server.verify.__annotations__["body"].model_validate(request))

        assert response.reward == 1.0
        assert response.model_extra["actual_handoff_tools"] == ["research_agent", "writer_agent"]
        assert response.model_extra["missing_handoff_tools"] == []
