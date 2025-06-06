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

# Adapted from https://github.com/vllm-project/vllm/blob/23c1b10a4c8cd77c5b13afa9242d67ffd055296b/Dockerfile
ARG CUDA_VERSION=12.3.0
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS builder
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.3.0
ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Set up ccache
ENV CCACHE_DIR=/ccache
RUN mkdir -p /ccache
ENV PATH="/usr/lib/ccache:${PATH}"

WORKDIR /app

# Copy and install build requirements first (for better caching)
COPY requirements-build.txt .
RUN python3 -m pip install --no-cache-dir -r requirements-build.txt

# Add the rest of the source code
COPY cmake ./cmake
COPY CMakeLists.txt .
COPY csrc ./csrc
COPY sllm_store ./sllm_store
COPY setup.py .
COPY pyproject.toml .
COPY MANIFEST.in .
COPY requirements.txt .
COPY README.md .
COPY proto ./proto

# Set build environment variables
ENV CMAKE_BUILD_TYPE=Release
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# Install additional build tools and build the wheel
RUN python3 -m pip install --no-cache-dir setuptools wheel twine && \
    python3 setup.py sdist bdist_wheel

# Output stage to extract the wheel
FROM alpine:latest AS output
COPY --from=builder /app/dist /dist