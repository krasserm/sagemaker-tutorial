#!/bin/bash -e

S3_PATH=$1
CACHE=.cache
ARCHIVE=cifar-10-python.tar.gz

if [ ! -f ${CACHE}/${ARCHIVE} ]
then
  mkdir -p ${CACHE}
  wget -O ${CACHE}/${ARCHIVE} https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
fi

# Upload CIFAR-10 dataset archive to user-defined S3 path
aws s3 cp ${CACHE}/${ARCHIVE} ${S3_PATH}/${ARCHIVE}
