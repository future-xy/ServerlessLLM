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
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages for wget and HTTPS
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates git

# Set the working directory
WORKDIR /app

RUN conda install python=3.10
RUN pip install -U pip

# Install checkpoint store
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
COPY sllm_store/cmake /app/sllm_store/cmake
COPY sllm_store/CMakeLists.txt /app/sllm_store/CMakeLists.txt
COPY sllm_store/csrc /app/sllm_store/csrc
COPY sllm_store/sllm_store /app/sllm_store/sllm_store
COPY sllm_store/setup.py /app/sllm_store/setup.py
COPY sllm_store/pyproject.toml /app/sllm_store/pyproject.toml
COPY sllm_store/MANIFEST.in /app/sllm_store/MANIFEST.in
COPY sllm_store/setup.cfg /app/sllm_store/setup.cfg
COPY sllm_store/requirements.txt /app/sllm_store/requirements.txt
COPY sllm_store/README.md /app/sllm_store/README.md
COPY sllm_store/proto /app/sllm_store/proto
RUN cd sllm_store && pip install .

COPY requirements.txt /app/
COPY requirements-worker.txt /app/
RUN pip install -r requirements.txt

COPY pyproject.toml setup.py setup.cfg py.typed /app/
COPY sllm/serve /app/sllm/serve
COPY sllm/cli /app/sllm/cli
COPY examples examples
COPY README.md /app/
RUN pip install .

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV NODE_TYPE=HEAD
# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
