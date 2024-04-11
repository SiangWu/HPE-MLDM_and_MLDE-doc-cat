import os
from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from determined.pytorch import DataLoader, PyTorchTrial
from tqdm import tqdm
from data import DogCatDataset, get_test_transforms, download_pach_repo
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class DogCatTrial(PyTorchTrial):
    def __init__(self, context):
        self.context = context
        self.download_directory = '/run/determined/workdir'
        self.data_dir = self.download_data()

        self.test_transform = get_test_transforms()
        self.model = self.context.wrap_model(
            self.build_model()
        )
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adadelta(
                self.model.parameters(), 
                lr=self.context.get_hparam('learning_rate')
            )
        )
        self.transform = transforms.Compose([
            transforms.Resize([230, 230]),
            transforms.RandomHorizontalFlip(p=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        print(f'data_dir: {self.data_dir}')


    def build_model(self) -> nn.Module:
        model = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1
        )  # Using ResNet-50
        model.fc = nn.Linear(
            in_features=2048, 
            out_features=2
        )  # for 2 classes

        return model
    

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = self.model(data)
        loss = torch.nn.functional.cross_entropy(
            output, 
            labels
        )

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {'loss': loss}


    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = self.model(data)
        validation_loss = torch.nn.functional.nll_loss(output, labels).item()

        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(data)

        return {validation_loss: validation_loss, 'accuracy': accuracy}


    def download_data(self) -> str:
        data_config = self.context.get_data_config()
        data_dir = os.path.join(self.download_directory)

        print(f'data_dir: {data_dir}')

        download_pach_repo(
            pachyderm_host=data_config['pachyderm']['host'],
            pachyderm_port=data_config['pachyderm']['port'],
            project=data_config['pachyderm']['project'],
            repo=data_config['pachyderm']['train_repo'],
            branch=data_config['pachyderm']['branch'],
            root_dir=data_dir
        )  # Download training data

        download_pach_repo(
            pachyderm_host=data_config['pachyderm']['host'],
            pachyderm_port=data_config['pachyderm']['port'],
            project=data_config['pachyderm']['project'],
            repo=data_config['pachyderm']['test_repo'],
            branch=data_config['pachyderm']['branch'],
            root_dir=data_dir
        )  # Download testing data

        return data_dir


    def build_train_dataset(self):
        ds = DogCatDataset(
            self.data_dir, 
            train=True, 
            transform=self.transform
        )

        return ds


    def build_test_dataset(self):
        ds = DogCatDataset(
            self.data_dir, 
            train=False
        )

        return ds


    def build_training_data_loader(self) -> Any:
        ds = self.build_train_dataset()

        return DataLoader(
            ds, 
            batch_size=self.context.get_per_slot_batch_size()
        )


    def build_validation_data_loader(self) -> Any:
        ds = self.build_test_dataset()

        return DataLoader(
            ds, 
            batch_size=self.context.get_per_slot_batch_size()
        )
