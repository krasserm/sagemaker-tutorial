import json
import os
import socket

from app import train


def main():
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = os.environ["SM_NETWORK_INTERFACE_NAME"]

    # List of nodes that participate in multi-node training.
    hosts = json.loads(os.environ["SM_HOSTS"])

    # Name and rank of current node
    host_c = os.environ['SM_CURRENT_HOST']
    rank_c = hosts.index(host_c)

    # Name and IP address of master node.
    host_0 = hosts[0]
    host_0_ip = socket.gethostbyname(host_0)

    # Set torch.distributed specific environment variables.
    os.environ["MASTER_ADDR"] = host_0_ip
    os.environ["MASTER_PORT"] = "29400"
    os.environ["WORLD_SIZE"] = str(len(hosts))
    os.environ["NODE_RANK"] = str(rank_c)

    # Call training script on current node
    train.main()


if __name__ == "__main__":
    main()
