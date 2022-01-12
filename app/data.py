from typing import Any
from pl_bolts.datamodules import cifar10_datamodule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision import transforms


@DATAMODULE_REGISTRY
class CIFAR10DataModule(cifar10_datamodule.CIFAR10DataModule):
    def __init__(self, num_workers: int = 1, val_split: int = 10000, normalize: bool = True, **kwargs: Any):
        super().__init__(num_workers=num_workers, val_split=val_split, normalize=normalize, **kwargs)
        self._extend_train_transforms()

    def _extend_train_transforms(self) -> None:
        """
        Add data augmentation (random crop/flip) to default transforms for training.
        """
        t = self.default_transforms()
        t.transforms.insert(0, transforms.RandomCrop(32, padding=4))
        t.transforms.insert(1, transforms.RandomHorizontalFlip())
        self.train_transforms = t
