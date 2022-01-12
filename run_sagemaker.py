import argparse

from sagemaker import LocalSession, Session
from sagemaker.estimator import Estimator


DEFAULT_HYPERPARAMS = {
    "data": "CIFAR10DataModule",
    "data.batch_size": "32",
    "optimizer": "Adam",
    "optimizer.lr": "0.001",
    "trainer.accelerator": "gpu",
    "trainer.devices": "-1",
    "trainer.max_epochs": "5",
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

    if local:
        session = LocalSession()
    else:
        session = Session()

    estimator = Estimator(image_uri=args.image_uri,
                          role=args.role,
                          instance_type=args.instance_type,
                          instance_count=args.instance_count,
                          output_path=args.output_path,
                          sagemaker_session=session,
                          hyperparameters=hyperparams)

    estimator.fit(inputs=args.input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_uri", default="sagemaker-tutorial")
    parser.add_argument("--role", default="arn:aws:iam::000000000000:role/dummy")
    parser.add_argument("--input_path", default="file://.cache")
    parser.add_argument("--output_path", default="file://output")
    parser.add_argument("--instance_type", default="local_gpu")
    parser.add_argument("--instance_count", default=1, type=int)
    parser.add_argument("--hyperparams", nargs="*", default=[])
    main(parser.parse_args())
