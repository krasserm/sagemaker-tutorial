# SageMaker tutorial

This repository contains the source code for articles 

- [Using AWS SageMaker with minimal dependencies, part 1 - Distributed model training with PyTorch Lightning](https://krasserm.github.io/2022/01/21/sagemaker-multi-node/) and 
- [Using AWS SageMaker with minimal dependencies, part 2 - Fault-tolerant model training on spot instances](https://krasserm.github.io/2022/02/26/sagemaker-fault-tolerance/)

and provides instructions for running the documented examples.

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/)

### SageMaker local mode

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Ability to [manage Docker as non-root-user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)

### SageMaker on AWS

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- [AWS SageMaker Domain onboarding](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)

## Training

Download CIFAR-10 data to local `.cache` directory with:

```bash
mkdir -p .cache
wget -O .cache/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

Build the `sagemaker-tutorial` Docker image:

```bash
./docker/build-image.sh
```

If you'd rather like to use the [torchrun based training script wrapper](app/train_multi_node_torchrun.py) 
(`app/train_multi_node_torchrun.py`) instead of the [default one](app/train_multi_node.py)
(`app/train_multi_node.py`) build the Docker image with:

```bash
./docker/build-image.sh --build-arg SAGEMAKER_PROGRAM=-app/train_multi_node_torchrun.py
```

### With SageMaker in local mode

Local single-node i.e. single-container training for 5 epochs using all available GPUs can then be started with:

```bash
python run_sagemaker.py
```

This uses the `DEFAULT_HYPERPARAMS` defined in [run_sagemaker.py](run_sagemaker.py) which are the same as in training
[without SageMaker](#without-sagemaker). 

Local multi-node i.e. multi-container training with 4 containers using the CPU can be started with:  

```bash
python run_sagemaker.py \
  --instance_type=local \
  --instance_count=4 \
  --hyperparams \
    trainer.accelerator=cpu \
    trainer.devices=1
```

The `--hyperparams` defined on the command line override the corresponding entries in `DEFAULT_HYPERPARAMS`. Additionally, 
you may want to set the `trainer.max_epochs=1` hyperparameter to reduce training time for testing purposes. 

If you don't have a valid `.aws` configuration in your home directory, you'll probably get a region error. In this case
set the `AWS_DEFAULT_REGION` environment variable to a valid region e.g.:

```bash
export AWS_DEFAULT_REGION=us-east-1
```

### With SageMaker in the cloud

For training with SageMaker in the cloud you need

1. a SageMaker execution role created during [onboarding](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)
2. an S3 path for reading training input data 
3. an S3 path for writing model data and tensorboard logs

The following commands use the environment variable `SAGEMAKER_ROLE` to refer to 1 and `S3_PATH` to refer to 2 and 3:

```bash
export SAGEMAKER=ROLE=arn:aws:iam::<account-id>:role/<role-name>
export S3_PATH=s3://<my-bucket>/<my-prefix>
```

Replace `<account-id>`, `<role-name>`, `<my-bucket>` and `<my-prefix>` with values appropriate for your AWS environment. 
Before training can be started, the `sagemaker-tutorial` Docker image must be pushed to the AWS Elastic Container 
Registry with:

```bash
./docker/push-image.sh
```

A private `sagemaker-tutorial` repository is created automatically if it doesn't exist yet. Next, upload the CIFAR-10 
dataset to S3:

```bash
./upload-cifar-10.sh ${S3_PATH}/datasets/cifar-10
```

Multi-node, multi-GPU training for 5 epochs on 2 `ml.g4dn.12xlarge` spot instances (4 GPUs each) can be started with:

```bash
JOB_NAME="sagemaker-tutorial-$(date '+%Y-%m-%d-%H-%M-%S')"; python run_sagemaker.py \
  --job_name=${JOB_NAME} \
  --image_uri=$(docker/image-uri.sh) \
  --role=${SAGEMAKER_ROLE} \
  --instance_type=ml.g4dn.12xlarge \
  --instance_count=2 \
  --input_path=${S3_PATH}/datasets/cifar-10 \
  --output_path=${S3_PATH}/output \
  --checkpoint_path=${S3_PATH}/output/${JOB_NAME}/checkpoints \
  --spot_instances=true \
  --max_retry=3 \
  --max_wait=14400 \
  --max_run=3600 \
  --hyperparams \
    trainer.max_epochs=5 \
    logger.save_dir=${S3_PATH}/output/${JOB_NAME}/logger-output \
    logger.flush_secs=10
```

`--max_run=3600` limits the maximum training time to 1 hour and `--max_wait=14400` limits the maximum waiting time for 
spot instances to 4 hours. Upon spot instance failures, training is restarted at most 3 times (`--max_retry=3`). Training
progress can be monitored during training with:

```bash
tensorboard --logdir ${S3_PATH}/tensorboard
```

### Without SageMaker in a conda environment

Calling `app/train.py` directly without using SageMaker requires an activated `sagemaker-tutorial` conda environment.

```bash
conda env create -f environment.yml
conda activate sagemaker-tutorial
```

Training for 5 epochs on all local GPUs can be started with:

```bash
PYTHONPATH=. python app/train.py \
  --data=CIFAR10DataModule \
  --data.data_dir=.cache \
  --optimizer=Adam \
  --optimizer.lr=1e-3 \
  --trainer.accelerator=gpu \
  --trainer.devices=-1 \
  --trainer.max_epochs=5 \
  --trainer.weights_save_path=logs/checkpoints \
  --logger.save_dir=logs/tensorboard \
  --logger.name=tutorial
```

### Without SageMaker in a Docker container 

To bypass the SageMaker training toolkit, the container is run with command `python app/train.py ...` directly, instead
of `train` (as SageMaker does).

```bash
docker run \
  -v $(pwd)/.cache:/opt/ml/code/.cache \
  -v $(pwd)/logs:/opt/ml/code/logs \
  --rm \
  --name app \
  --runtime=nvidia \
  sagemaker-tutorial:latest \
  python app/train.py \
    --data=CIFAR10DataModule \
    --data.data_dir=.cache \
    --optimizer=Adam \
    --optimizer.lr=1e-3 \
    --trainer.accelerator=gpu \
    --trainer.devices=-1 \
    --trainer.max_epochs=1 \
    --trainer.weights_save_path=logs/checkpoints \
    --logger.save_dir=logs/tensorboard \
    --logger.name=tutorial
```
