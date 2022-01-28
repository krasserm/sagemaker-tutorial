#!/bin/bash -e

docker build -f docker/Dockerfile -t sagemaker-tutorial . $@
