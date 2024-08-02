# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import uuid
from typing import Dict, Optional

import ray

from serverless_llm.serve.logger import init_logger

# from serverless_llm.serve.inference_instance import start_instance

from ..utils import InstanceStatus
from .roundrobin_router import RoundRobinRouter

logger = init_logger(__name__)


class MigrationRouter(RoundRobinRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
    ) -> None:
        super().__init__(
            model_name, resource_requirements, backend, backend_config
        )

    async def execute_migration_plan(self, migration_plan):
        self.log(f"Executing migration plan: {migration_plan}")
        source_instance_id = migration_plan.source_instance_id
        target_node_id = migration_plan.target_node_id
        # start the instance on the target node
        startup_config = {
            "num_cpus": 1,
            "num_gpus": 1,  # FIXME
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{target_node_id}": 0.1,
            },
        }
        target_instance_id = await self._create_instance(startup_config)
        # stop the instance on the source node
        await self._stop_instance(source_instance_id)
        return target_instance_id

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        async with self.instance_management_lock:
            instance = self.instances[instance_id]
            instance_status = await instance.get_status()
            instance_status.model_name = self.model_name
            return instance_status
