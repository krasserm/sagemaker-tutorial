import argparse

from sagemaker import LocalSession, Session
from sagemaker.estimator import Estimator


SM_CHECKPOINT_DIR = "/opt/ml/checkpoints"

DEFAULT_HYPERPARAMS = {
    "data": "CIFAR10DataModule",
    "data.batch_size": 32,
    "optimizer": "Adam",
    "optimizer.lr": 1e-3,
    "trainer.accelerator": "gpu",
    "trainer.devices": -1,
    "trainer.max_epochs": 5,
    "logger.name": "tutorial"
}


def parse_hyperparams(args):
    hyperparams = {}
    for param in args.hyperparams:
        key, value = param.split("=")
        hyperparams[key] = value
    return hyperparams


def main(args):
    local = args.instance_type.startswith("local")

    # merge dicts (Python 3.9+)
    hyperparams = DEFAULT_HYPERPARAMS | parse_hyperparams(args)

    # Extra environment variables for training script
    environment = None

    if args.checkpoint_path:
        environment = {"SM_CHECKPOINT_DIR": SM_CHECKPOINT_DIR}

    if local:
        session = LocalSession()
    else:
        session = Session()

    # Setting environment does not work in SageMaker in local mode
    # (See https://github.com/aws/sagemaker-python-sdk/issues/2930)
    estimator = Estimator(image_uri=args.image_uri,
                          role=args.role,
                          instance_type=args.instance_type,
                          instance_count=args.instance_count,
                          output_path=args.output_path,
                          checkpoint_s3_uri=args.checkpoint_path,
                          checkpoint_local_path=SM_CHECKPOINT_DIR,
                          use_spot_instances=args.spot_instances,
                          max_retry_attempts=args.max_retry,
                          max_wait=args.max_wait,
                          max_run=args.max_run,
                          sagemaker_session=session,
                          hyperparameters=hyperparams,
                          environment=environment)

    estimator.fit(inputs=args.input_path, job_name=args.job_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", default=None)
    parser.add_argument("--image_uri", default="sagemaker-tutorial")
    parser.add_argument("--role", default="arn:aws:iam::000000000000:role/dummy")
    parser.add_argument("--input_path", default="file://.cache")
    parser.add_argument("--output_path", default="file://output")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--instance_type", default="local_gpu")
    parser.add_argument("--instance_count", default=1, type=int)
    parser.add_argument("--spot_instances", default=False, type=bool)
    parser.add_argument("--max_run", default=24 * 60 * 60, type=int)
    parser.add_argument("--max_wait", default=36 * 60 * 60, type=int)
    parser.add_argument("--max_retry", default=5, type=int)
    parser.add_argument("--volume_size", default=30, type=int)
    parser.add_argument("--hyperparams", nargs="*", default=[])
    main(parser.parse_args())
