#!/bin/bash

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)

IMAGE_NAME=sagemaker-tutorial
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:latest"

aws ecr describe-repositories --repository-names "${IMAGE_NAME}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
  echo "Create repository ${IMAGE_NAME}"
  aws ecr create-repository --repository-name "${IMAGE_NAME}" > /dev/null
fi

echo 'Login to AWS ECR'
aws ecr get-login-password --region ${REGION} | \
docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo 'Build Docker image'
docker/build-image.sh $@

echo 'Push Docker image'
docker tag ${IMAGE_NAME} ${IMAGE_URI}
docker push ${IMAGE_URI}
