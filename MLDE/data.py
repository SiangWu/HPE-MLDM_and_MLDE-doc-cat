import os
from PIL import Image
import subprocess
from typing import Any, Dict, Tuple
import torch
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import time

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize(
            [230, 230]
        ),
        transforms.RandomHorizontalFlip(
            p=0.05
        ),
        transforms.RandomRotation(
            degrees=10
        ),
        transforms.RandomCrop(
            [224, 224]
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), 
            (0.229, 0.224, 0.225)
        )
    ])


class DogCatDataset(Dataset):
    """
    Dogs-vs-Cats dataset.
    """

    def __init__(self, root_dir, train, transform=None):
        file_dir = 'train' if train else 'test'
        self.file_path = os.path.join(
            root_dir, 
            file_dir
        )
        print(f'Dataset path: {self.file_path}')

        self.files = [f for f in os.listdir(self.file_path) if f.endswith('.jpg')]
        self.transform = transform


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = io.imread(
            os.path.join(
                self.file_path, 
                img_name
            )
        )
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = 0 if img_name.startswith('dog') else 1  # 檔名是 'dog' 開頭的都標記為 0，反之為 1

        return (image, label)


def download_pach_repo(pachyderm_host, pachyderm_port, project, repo, branch, root_dir):
    subprocess.run([f'pachctl config update context --pachd-address {pachyderm_host}:{pachyderm_port}'], shell=True)
    subprocess.run([f'pachctl config update context --project {project}'], shell=True)
    subprocess.run(['pachctl version'], shell=True)
    subprocess.run([f'pachctl get file -r {repo}@{branch}:/{repo} --output {root_dir}/{repo}'], shell=True)  # 下載資料

    subprocess.run(['ls -al'], shell=True)
