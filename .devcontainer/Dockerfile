# Use the base image of Ubuntu
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    gnupg \
    software-properties-common \
    vim \
    sudo \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Make 'python' and 'pip' available
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install Zig (replace the version with the desired one)
ENV ZIG_VERSION=0.15.2
RUN wget https://ziglang.org/download/${ZIG_VERSION}/zig-linux-$(arch)-${ZIG_VERSION}.tar.xz \
    && tar xf zig-linux-$(arch)-${ZIG_VERSION}.tar.xz \
    && mv zig-linux-$(arch)-${ZIG_VERSION} /opt/zig \
    && ln -s /opt/zig/zig /usr/local/bin/zig \
    && rm zig-linux-$(arch)-${ZIG_VERSION}.tar.xz

# Install Python packages needed by helper scripts
# (onnx is the key one; numpy/protobuf are common deps; onnxruntime is handy for local checks)
RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    pip install --no-cache-dir onnx numpy protobuf onnxruntime os onnxsim

# Set the working directory
WORKDIR /workspace

# Set a non-root user (optional but recommended)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

CMD [ "/bin/bash" ]
