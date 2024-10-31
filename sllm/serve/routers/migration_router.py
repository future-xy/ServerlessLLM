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
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=10,
            num_gpu=self.resource_requirements["num_gpus"],
        )

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
        logger.info(
            f"Initialized instance {instance_id} for model {self.model_name}"
        )
        await instance.backend_instance.init_backend.remote()
        logger.info(
            f"Initialized backend for instance {instance_id} for model {self.model_name}"
        )
        # stop the instance on the source node
        source_instance = self.ready_instances[source_instance_id].backend_instance
        migration_iter = 0
        while True:
            logger.info(f"Migration iteration {migration_iter}")
            current_tokens = await source_instance.get_current_tokens.remote()
            if not current_tokens or len(current_tokens) <= 10:
                logger.info(
                    "Migration completed:"
                    f"{None if not current_tokens else len(current_tokens)} tokens"
                )
                break
            instance.backend_instance.resume_kv_cache.remote(
                current_tokens
            )
            migration_iter += 1
            logger.info(
                f"Migration iteration {migration_iter} completed: {current_tokens}"
            )

        # # TODO: make the two steps as atomic
        # async with self.instance_management_lock:
        #     self.ready_instances[instance_id] = instance
        # await self._shutdown_instance(source_instance_id)
        logger.info(
            f"Migrated instance {source_instance_id} to {instance_id}"
        )
        async with self.instance_management_lock:
            if source_instance_id not in self.ready_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.ready_instances.pop(source_instance_id)
            async with instance.lock:
                instance.status = False
            self.ready_instances[instance_id] = instance
        await instance.backend_instance.shutdown.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, source_instance_id, self.resource_requirements
        )
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
