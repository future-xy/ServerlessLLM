# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

import asyncio
import copy
from abc import ABCMeta
from dataclasses import dataclass
from typing import List, Mapping, Optional

import ray

from sllm.serve.logger import init_logger

from ..utils import InstanceStatus
from .fcfs_scheduler import FcfsScheduler

logger = init_logger(__name__)


@dataclass
class MigrationPlan:
    migration_time: float
    target_model: str
    source_node_id: int
    source_instance_id: int
    target_node_id: int


@dataclass
class AllocationPlan:
    node_id: int
    latency: float
    migration_plans: Optional[List[MigrationPlan]] = None


class StorageAwareScheduler(FcfsScheduler):
    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__(scheduler_config)

        self.enable_migration = scheduler_config.get("enable_migration", False)

        self.store_manager = None

        self.model_scheduler_config = {}

    async def _control_loop(self):
        logger.info("Starting control loop")
        while self.running:
            loading_requests = []
            logger.info(f"Loading requests: {loading_requests}")
            async with self.queue_lock:
                for (
                    model_name,
                    loading_queue,
                ) in self.model_loading_queues.items():
                    for idx, (
                        request_time,
                        num_gpus,
                        allocation_result,
                    ) in enumerate(loading_queue):
                        loading_requests.append(
                            (
                                model_name,
                                idx,
                                request_time,
                                num_gpus,
                                allocation_result,
                            )
                        )
            if loading_requests:
                logger.info(f"Loading requests are: {loading_requests}")
            if self.store_manager is None:
                try:
                    self.store_manager = ray.get_actor("store_manager")
                except ValueError:
                    logger.error("Store manager not found")
                    await asyncio.sleep(1)
                    continue
            # first come first serve
            if len(loading_requests) > 0:
                worker_nodes = await self._get_worker_nodes()
                logger.info(f"Worker nodes: {worker_nodes}")
                model_info = await self.store_manager.get_model_info.remote()
                logger.info(f"Model info: {model_info}")
                store_info = await self.store_manager.get_store_info.remote()
                logger.info(f"Store info: {store_info}")
                hardware_info = (
                    await self.store_manager.get_hardware_info.remote()
                )
                logger.info(f"Hardware info: {hardware_info}")
                loading_requests.sort(key=lambda x: x[1])
                logger.info(f"Sorted loading requests: {loading_requests}")
                for (
                    model_name,
                    idx,
                    request_time,
                    num_gpus,
                    allocation_result,
                ) in loading_requests:
                    logger.info(f"Processing request for model {model_name}")
                    scheduling_options = await self.schedule(
                        model_name,
                        num_gpus,
                        worker_nodes,
                        model_info,
                        store_info,
                        hardware_info,
                    )
                    # sort by latency
                    if scheduling_options:
                        scheduling_options.sort(
                            key=lambda x: (x.latency, x.node_id)
                        )
                        logger.info(
                            f"Sorted scheduling options: {scheduling_options}"
                        )
                        allocation_plan = scheduling_options[0]
                        if allocation_plan.migration_plans is not None:
                            # execute migration plans
                            for (
                                migration_plan
                            ) in allocation_plan.migration_plans:
                                target_model = migration_plan.target_model
                                target_request_router = ray.get_actor(
                                    target_model, namespace="models"
                                )
                                logger.info(
                                    f"Executing migration plan: {migration_plan}"
                                )
                                target_node_id = migration_plan.target_node_id
                                worker_nodes[target_node_id]["free_gpu"] -= (
                                    num_gpus
                                )
                                target_instance_id = await target_request_router.execute_migration_plan.remote(
                                    migration_plan
                                )
                                if target_instance_id is None:
                                    logger.info(
                                        f"Failed to execute migration plan: {migration_plan}"
                                    )
                                    worker_nodes[target_node_id][
                                        "free_gpu"
                                    ] += num_gpus
                                else:
                                    logger.info(
                                        f"Migrated instance {target_model} to node {target_node_id} instance {target_instance_id}"
                                    )

                        node_id = allocation_plan.node_id
                        async with self.queue_lock:
                            self.model_loading_queues[model_name].pop(idx)
                            logger.info(
                                f"Allocated node {node_id} for model {model_name}"
                            )
                            allocation_result.set_result(node_id)

                        worker_nodes[node_id]["free_gpu"] -= num_gpus
                        await self.store_manager.load_to_host.remote(
                            node_id, model_name
                        )
                    else:
                        logger.info(f"No available node for model {model_name}")
                await self._update_worker_nodes(worker_nodes)

            await asyncio.sleep(1)

    async def schedule(
        self,
        model_name,
        num_gpus,
        worker_nodes,
        model_info,
        store_info,
        hardware_info,
    ) -> List[AllocationPlan]:
        scheduling_options = []
        logger.info(f"Checking model {model_name}")
        for node_id, node_info in worker_nodes.items():
            logger.info(f"Checking node {node_id}, node info: {node_info}")
            if node_id not in store_info:
                logger.error(f"Node {node_id} not found in store info")
                continue
            free_gpu = node_info["free_gpu"]
            logger.info(f"Node {node_id} has {free_gpu} free GPUs")
            (
                node_store_info,
                pinned_memory_pool,
                node_waiting_time,
            ) = store_info[node_id]
            if model_name not in node_store_info:
                logger.info(f"Model {model_name} not found in node {node_id}")
                # Note(Yao): Downloading from HuggingFace Hub is
                # slower than network bandwidth and difficult to estimate.
                # So we just consider local checkpoints for now.
                continue
            latency = self._get_model_loading_time(
                model_name,
                model_info[model_name],
                hardware_info[node_id],
                node_waiting_time,
                pinned_memory_pool,
            )
            if free_gpu >= num_gpus:
                scheduling_options.append(AllocationPlan(node_id, latency))
            elif self.enable_migration:
                gpu_shortage = num_gpus - free_gpu
                logger.info(
                    f"Node {node_id} does not have enough GPU, trying migration"
                )
                migration_plans = await self.get_migration_plans(
                    model_name,
                    gpu_shortage,
                    node_id,
                    copy.deepcopy(worker_nodes),
                    copy.deepcopy(model_info),
                    copy.deepcopy(store_info),
                    copy.deepcopy(hardware_info),
                )
                if migration_plans is not None:
                    migration_latency = max(
                        [plan.migration_time for plan in migration_plans]
                    )
                    scheduling_options.append(
                        AllocationPlan(
                            node_id=node_id,
                            latency=latency + migration_latency,
                            migration_plans=migration_plans,
                        )
                    )
            else:
                logger.info(
                    f"Node {node_id} does not have enough GPU and migration is disabled"
                )
        return scheduling_options

    async def get_migration_plans(
        self,
        model_name: str,
        gpu_shortage: int,
        source_node_id: int,
        worker_nodes: Mapping,
        model_info: Mapping,
        store_info: Mapping,
        hardware_info: Mapping,
    ) -> Optional[List[MigrationPlan]]:
        released_gpu = 0
        migrated_instances = []
        migration_plans = []
        migratable_instances = {}
        request_routers = {}
        async with self.metadata_lock:
            logger.info(f"Checking migratable instances for model {model_name}")
            for target_model_name in self.model_instance:
                # Skip the instances that is already running the model
                if target_model_name == model_name:
                    continue
                for instance_id, node_id in self.model_instance[
                    target_model_name
                ].items():
                    if node_id == source_node_id:
                        logger.info(
                            f"Checking instance {instance_id} of model {target_model_name}"
                        )
                        if target_model_name not in request_routers:
                            request_routers[target_model_name] = ray.get_actor(
                                target_model_name, namespace="models"
                            )
                        logger.info(
                            f"Getting status for instance {instance_id} of model {target_model_name}"
                        )
                        instance_status = await request_routers[
                            target_model_name
                        ].get_instance_status.remote(instance_id)
                        if instance_status:
                            logger.info(
                                f"Instance {instance_id} status: {instance_status}"
                            )
                            migratable_instances[instance_id] = instance_status
        logger.info(f"Migratable instances: {migratable_instances}")
        for instance_id, instance in migratable_instances.items():
            # Try to migrate this instance to another node
            for node_id, node_info in worker_nodes.items():
                if node_id == source_node_id:
                    continue
                if node_info["free_gpu"] >= instance.num_gpu:
                    loading_time = self._get_model_loading_time(
                        instance.model_name,
                        model_info[instance.model_name],
                        hardware_info[node_id],
                        store_info[node_id][2],
                        store_info[node_id][1],
                    )
                    alpha = self.model_scheduler_config.get("alpha", 0.01)
                    beta = self.model_scheduler_config.get("beta", 0.1)
                    num_current_tokens = instance.num_current_tokens
                    resuming_time = alpha * num_current_tokens + beta
                    migration_time = loading_time + resuming_time
                    migrated_instances.append(instance_id)
                    migration_plans.append(
                        MigrationPlan(
                            migration_time=migration_time,
                            target_model=instance.model_name,
                            source_node_id=source_node_id,
                            source_instance_id=instance_id,
                            target_node_id=node_id,
                        )
                    )
                    store_info[node_id][2] += loading_time
                    node_info["free_gpu"] -= instance.num_gpu
                    released_gpu += instance.num_gpu
                    break

            if released_gpu >= gpu_shortage:
                return migration_plans

        return None

    async def mark_resource(
        self, model_name: str, instance_id: str, node_id: int
    ) -> bool:
        logger.info(f"Model {model_name} instance {instance_id} marked")
        async with self.metadata_lock:
            if model_name not in self.model_instance:
                self.model_instance[model_name] = {}
            self.model_instance[model_name][instance_id] = node_id
        return node_id

    async def set_model_scheduler_config(
        self, model_name: str, scheduler_config: Mapping
    ) -> bool:
        logger.info(f"Setting scheduler config for model {model_name}")
        async with self.metadata_lock:
            self.model_scheduler_config[model_name] = scheduler_config
        return True

    def _get_model_loading_time(
        self,
        model_name: str,
        model_size: int,
        hardware_info: Mapping,
        node_waiting_time: float,
        pinned_memory_pool: Mapping,
    ) -> float:
        latency = 0
        if model_name not in pinned_memory_pool:
            latency += (
                node_waiting_time + model_size / hardware_info["disk_bandwidth"]
            )
            logger.info(f"Loading model {model_name} takes {latency} seconds")
        else:
            latency += model_size / hardware_info["pcie_bandwidth"]
            logger.info(f"Loading model {model_name} takes {latency} seconds")
        return latency
