import json
import os
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser

from app.model import ResNet18


class CLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        self.sm_training_data_dir = os.environ.get("SM_CHANNEL_TRAINING")
        self.sm_output_data_dir = os.environ.get("SM_OUTPUT_DATA_DIR")
        self.sm_model_dir = os.environ.get("SM_MODEL_DIR")
        self.sm_hosts = os.environ.get("SM_HOSTS", "[\"localhost\"]")
        self.num_nodes = len(json.loads(self.sm_hosts))
        super().__init__(*args, **kwargs)

    def before_instantiate_classes(self) -> None:
        if self.sm_training_data_dir:
            # Update config (instead of setting parser defaults) because
            # data module class is set dynamically as command line option.
            self.config["data"]["init_args"]["data_dir"] = self.sm_training_data_dir

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        # Bind num_classes property of the data module to model's num_classes parameter.
        parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")

        # Make TensorBoardLogger configurable under the "logger" namespace and
        # expose flush_secs keyword argument as additional command line option.
        parser.add_class_arguments(TensorBoardLogger, "logger")
        parser.add_argument("--logger.flush_secs", default=60, type=int)

        if self.sm_output_data_dir:
            parser.set_defaults({
                "trainer.weights_save_path": os.path.join(self.sm_output_data_dir, "checkpoints"),
                "logger.save_dir": os.path.join(self.sm_output_data_dir, "tensorboard")
            })

    def instantiate_trainer(self, **kwargs):
        # Instantiate trainer with configured logger and number of nodes as arguments.
        return super().instantiate_trainer(logger=self.config_init["logger"], num_nodes=self.num_nodes, **kwargs)


def main():
    trainer_defaults = {
        # Trainer default configuration is defined in file app/trainer.yaml.
        "default_config_files": [os.path.join("app", "trainer.yaml")]
    }

    # Instantiate trainer, model and data module.
    cli = CLI(model_class=ResNet18, parser_kwargs=trainer_defaults, save_config_overwrite=True, run=False)

    # Run training and validation.
    cli.trainer.fit(cli.model, cli.datamodule)

    if cli.trainer.is_global_zero and cli.sm_model_dir:
        # Load best checkpoint.
        ckpt_path = cli.trainer.checkpoint_callback.best_model_path
        ckpt = ResNet18.load_from_checkpoint(ckpt_path)

        # Write best model to SageMaker model directory.
        model_path = os.path.join(cli.sm_model_dir, "model.pt")
        torch.save(ckpt.model.state_dict(), model_path)

        # Checkpoint not needed (yet), delete it.
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
