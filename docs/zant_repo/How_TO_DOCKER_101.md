# Z-Ant Project - Quick Start Guide with Docker

Welcome to the Z-Ant project! This guide will help you set up a Docker-based development environment, allowing for a consistent setup across all contributors. You'll learn how to build and run the Docker container, manage containers, and integrate with Visual Studio Code.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
   - [Clone the Repository](#clone-the-repository)
   - [Switch to the Dockerization Branch](#switch-to-the-dockerization-branch)
   - [Build the Docker Image](#build-the-docker-image)
3. [Managing Docker Containers](#managing-docker-containers)
   - [List Docker Images](#list-docker-images)
   - [Run the Docker Container](#run-the-docker-container)
   - [Access the Running Container](#access-the-running-container)
   - [List Running Containers](#list-running-containers)
   - [Stop a Running Container](#stop-a-running-container)
4. [Using Visual Studio Code with Docker](#using-visual-studio-code-with-docker)
   - [Install Necessary Extensions](#install-necessary-extensions)
   - [Open the Project in a Container](#open-the-project-in-a-container)
   - [Develop Inside the Container](#develop-inside-the-container)
5. [Updating the Docker Environment](#updating-the-docker-environment)
6. [Additional Docker Commands](#additional-docker-commands)
7. [Notes](#notes)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Docker**: Install Docker Desktop from [Docker's official website](https://www.docker.com/get-started).
- **Visual Studio Code (VSCode)**: Download and install from [here](https://code.visualstudio.com/).
- **Git**: Ensure Git is installed for version control operations.

## Getting Started

### Clone the Repository

Clone the Z-Ant repository to your local machine:

```bash
git clone https://github.com/ZIGTinyBook/Z-Ant.git
cd Z-Ant
```

### Switch to the Dockerization Branch

Switch to the `dockerization` branch to access the Docker setup:

```bash
git checkout dockerization
```

### Build the Docker Image

Build the Docker image using the provided Dockerfile:

```bash
docker build -f .devcontainer/Dockerfile -t z-ant-dev .
```

- `-f .devcontainer/Dockerfile`: Specifies the path to the Dockerfile.
- `-t z-ant-dev`: Tags the image with the name `z-ant-dev`.

## Managing Docker Containers

### List Docker Images

To verify that the Docker image was created successfully:

```bash
docker images
```

You should see `z-ant-dev` listed among the images.

### Run the Docker Container

Run the container in interactive mode:

```bash
docker run -it --name z-ant-container -v "$(pwd):/workspace" z-ant-dev /bin/bash
```

- `-it`: Runs the container in interactive mode with a TTY.
- `--name z-ant-container`: Names the container for easier reference.
- `-v "$(pwd):/workspace"`: Mounts the current directory into `/workspace` inside the container.
- `/bin/bash`: Starts a bash shell inside the container.

### Access the Running Container

If you need to attach to the container again (in case you detached without stopping it):

```bash
docker exec -it z-ant-container /bin/bash
```

### List Running Containers

To list all running containers:

```bash
docker ps
```

To list all containers, including stopped ones:

```bash
docker ps -a
```

### Stop a Running Container

To stop the container:

```bash
docker stop z-ant-container
```

To remove the container after stopping it:

```bash
docker rm z-ant-container
```

## Using Visual Studio Code with Docker

### Install Necessary Extensions

In VSCode, install the following extensions:

- **Remote - Containers**: [Visual Studio Marketplace Link](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- **Zig Language Support**: [Visual Studio Marketplace Link](https://marketplace.visualstudio.com/items?itemName=ziglang.vscode-zig)

### Open the Project in a Container

1. **Open VSCode in the Project Directory**:

   From your terminal:

   ```bash
   code .
   ```

2. **Reopen in Container**:

   - Press `F1` or `Ctrl+Shift+P` to open the Command Palette.
   - Type `Remote-Containers: Reopen in Container` and select it.
   - VSCode will build the Docker image (if not already built) and reopen the workspace inside the container.

### Develop Inside the Container

- **Integrated Terminal**: Use the terminal in VSCode; it runs inside the container.
- **Build and Run**:

  ```bash
  zig build run
  ```

- **Debugging**: Set breakpoints and debug your application as usual.
- **Extensions**: Install any additional VSCode extensions you need; they will be installed inside the container.

## Updating the Docker Environment

When you pull updates that include changes to the Dockerfile or `.devcontainer` configurations, you need to rebuild the Docker image.

### Pull the Latest Changes

```bash
git pull
```

### Rebuild the Docker Image

If you're using VSCode:

1. **Rebuild Container**:

   - Press `F1` or `Ctrl+Shift+P`.
   - Type `Remote-Containers: Rebuild and Reopen in Container` and select it.
   - VSCode will rebuild the image with the latest configurations.

Alternatively, rebuild manually:

```bash
docker build -f .devcontainer/Dockerfile -t z-ant-dev .
```

## Additional Docker Commands

### Remove Unused Containers and Images

- **Remove All Stopped Containers**:

  ```bash
  docker container prune
  ```

- **Remove All Unused Images**:

  ```bash
  docker image prune -a
  ```

### Inspect a Container

To view detailed information about a container:

```bash
docker inspect z-ant-container
```

### View Container Logs

If your container runs a service and you want to see the logs:

```bash
docker logs z-ant-container
```

## Notes

- **Workspace Synchronization**: The project directory is synchronized between your local machine and the container. Changes are reflected immediately.
- **Non-Root User**: The container uses a non-root user `vscode` for security.
- **Persisting Data**: Data outside the mounted workspace will not persist after the container is removed.
- **Environment Consistency**: Using Docker ensures that all contributors work in the same environment, reducing "it works on my machine" issues.

## Troubleshooting

### Permissions Issues

If you encounter file permission issues:

- Ensure that the `USER_UID` and `USER_GID` in the Dockerfile match your local user's UID and GID.
- Modify the Dockerfile's user creation section:

  ```dockerfile
  ARG USER_UID=your_uid
  ARG USER_GID=your_gid
  ```

- Rebuild the Docker image after making changes.

### Docker Build Errors

- **Network Issues**: Ensure you have a stable internet connection during the build process.
- **Docker Version**: Verify that you are using a recent version of Docker.

### VSCode Remote Containers Issues

- **Extension Installation**: Make sure the Remote - Containers extension is installed and enabled.
- **Container Rebuild**: If the container doesn't start correctly, try rebuilding it:

  - Press `F1` or `Ctrl+Shift+P`.
  - Select `Remote-Containers: Rebuild Container`.

### Zig Version Mismatch

If you need a different version of Zig:

- Modify the `ZIG_VERSION` in the Dockerfile:

  ```dockerfile
  ENV ZIG_VERSION=desired_version
  ```

- Update the download URL accordingly.
- Rebuild the Docker image.

---

Happy coding!

If you have any questions or need further assistance, feel free to open an issue or reach out to the maintainers.
