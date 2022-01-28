FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    python3.8-dev \
    build-essential \
    wget

ARG CONDA_VERSION=latest

# Install latest version of Miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# Location of application code as required by SageMaker.
ARG CODEPATH=/opt/ml/code

# Create sagemaker-tutorial conda environment.
COPY environment.yml $CODEPATH/environment.yml
RUN conda update -n base -c defaults conda
RUN conda env create -f $CODEPATH/environment.yml

# Copy application files.
COPY app $CODEPATH/app/

# Expose training script to SageMaker and support overriding
# at build-time. This can either be app/train_multi_node.py
# (default) or app/train_multi_node_torchrun.py.
ARG SAGEMAKER_PROGRAM=app/train_multi_node.py
ENV SAGEMAKER_PROGRAM=${SAGEMAKER_PROGRAM}

# Make all local GPUs visible
ENV NVIDIA_VISIBLE_DEVICES="all"

# Set working directory to root directory of application code
WORKDIR $CODEPATH

ENTRYPOINT ["app/train.sh"]
