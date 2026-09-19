"""Shared InfiniBand/RDMA passthrough for container engines.

Detection is presence-based: if the host exposes /dev/infiniband, the device
nodes are passed into the container so distributed backends can reach the RDMA
NIC. Hosts without InfiniBand get no extra flags.
"""

from __future__ import annotations

from pathlib import Path

from .engines import ContainerEngine

RDMA_DEVICE_PATH = "/dev/infiniband"
RDMA_MEMLOCK_ULIMIT = "memlock=-1"


def host_rdma_device_nodes(rdma_path: str | Path = RDMA_DEVICE_PATH) -> list[Path]:
    """Return the InfiniBand device nodes present on the host."""
    path = Path(rdma_path)
    if not path.is_dir():
        return []
    try:
        return sorted(path.iterdir())
    except OSError:
        return []


def container_rdma_args(
    engine: str | ContainerEngine,
    rdma_path: str = RDMA_DEVICE_PATH,
) -> list[str]:
    """Return engine flags that expose the host InfiniBand devices to a container.

    Podman accepts the device directory, so it gets the directory plus the rdma
    supplementary group and an unlimited memlock ulimit. Docker cannot take a
    directory, so each device node is passed individually.
    """
    engine_name = engine.value if isinstance(engine, ContainerEngine) else engine
    if not Path(rdma_path).is_dir():
        return []

    if engine_name == ContainerEngine.PODMAN.value:
        return [
            "--device", rdma_path,
            "--group-add", "rdma",
            "--ulimit", RDMA_MEMLOCK_ULIMIT,
        ]

    if engine_name == ContainerEngine.DOCKER.value:
        args: list[str] = []
        for device in host_rdma_device_nodes(rdma_path):
            args.extend(["--device", str(device)])
        if args:
            args.extend(["--ulimit", RDMA_MEMLOCK_ULIMIT])
        return args

    return []
