import json
import os
import sys
import socket

from torch.distributed.run import parse_args, config_from_args
from torch.distributed.launcher.api import elastic_launch


def main():
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = os.environ["SM_NETWORK_INTERFACE_NAME"]

    # List of nodes that participate in multi-node training.
    hosts = json.loads(os.environ["SM_HOSTS"])

    # Name and IP address of master node.
    host_0 = hosts[0]
    host_0_ip = socket.gethostbyname(host_0)

    # job_id is used to identify the group of nodes
    job_id = os.environ["TRAINING_JOB_NAME"]

    n_gpus = os.environ["SM_NUM_GPUS"]
    n_procs = 1 if n_gpus == "0" else "gpu"

    args = ["--rdzv_backend=c10d",
            f"--rdzv_id={job_id}",
            f"--rdzv_endpoint={host_0_ip}:29400",
            f"--nnodes={len(hosts)}",
            f"--nproc_per_node={n_procs}",
            "app/train.py"] + sys.argv[1:]

    args = parse_args(args)
    config, ts, ts_args = config_from_args(args)
    elastic_launch(config=config, entrypoint=ts)(*ts_args)


if __name__ == "__main__":
    main()
