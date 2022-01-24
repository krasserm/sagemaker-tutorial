#!/bin/bash --login

conda activate sagemaker-tutorial
export PYTHONPATH=.

$@
