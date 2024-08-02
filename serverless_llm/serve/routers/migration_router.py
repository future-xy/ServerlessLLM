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

from ..utils import InstanceStatus, InstanceHandle
from .roundrobin_router import RoundRobinRouter
from ..inference_instance import start_instance

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
        logger.info(f"Executing migration plan: {migration_plan}")
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
        logger.info(f"Startup config: {startup_config}, {self.backend_config}")

        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # TODO: Add max_queue_length to instance
        instance = InstanceHandle(instance_id=instance_id, max_queue_length=10, 
                                  num_gpu=self.resource_requirements["num_gpus"])
        
        await start_instance.options(
            resources={
                "worker_node": 0.1,
                f"worker_id_{target_node_id}": 0.1,
            }
        ).remote(instance_id, self.backend, self.backend_config, startup_config)
        logger.info(
            f"Started instance {instance_id} for model {self.model_name}"
        )
        instance.backend_instance = ray.get_actor(instance_id)
        async with instance.lock:
            instance.ready = True
            instance.node_id = target_node_id
        await instance.backend_instance.init_backend.remote()
        async with self.instance_management_lock:
            self.ready_instances[instance_id] = instance

        # stop the instance on the source node
        await self._stop_instance(source_instance_id)
        return instance_id

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        logger.info(f"Getting status for instance: {instance_id}")
        async with self.instance_management_lock:
            if instance_id not in self.ready_instances:
                logger.info(f"Instance {instance_id} not found")
                return None
            instance = self.ready_instances[instance_id]
            logger.info(f"Instance: {instance}")
            instance_status = await instance.get_status()
            instance_status.model_name = self.model_name
            logger.info(f"Instance status: {instance_status}")
            return instance_status
