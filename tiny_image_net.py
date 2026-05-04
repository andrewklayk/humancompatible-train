import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
from tinyimagenet import TinyImageNet
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from fairret.statistic import PositiveRate, TruePositiveRate, FalsePositiveRate, PositivePredictiveValue, FalseOmissionRate
from fairret.loss import NormLoss
from humancompatible.train.fairness.utils import BalancedBatchSampler
import torch

def train_tinyimagenet():

    # define batch size here
    train_batch_size=128
    batch_size = 256
    
    # define the path here 
    dataset_path="~/.torchvision/tinyimagenet/"


    # define transforms function
    normalize_transform = T.Compose([ T.ToTensor(),
                                    T.Normalize(mean=TinyImageNet.mean,
                                std=TinyImageNet.std),
                                # Converting cropped images to tensors
    ])
    train_transform = T.Compose([ T.Resize(256), # Resize images to 256 x 256
                    T.CenterCrop(224), # Center crop image
                    T.RandomHorizontalFlip(),
                    normalize_transform

                    ])


    # load the data
    train = TinyImageNet(Path(dataset_path),split="train",transform=train_transform,imagenet_idx=True)
    val = TinyImageNet(Path(dataset_path),split="val",transform=normalize_transform,imagenet_idx=True)
    test = TinyImageNet(Path(dataset_path),split="test",transform=normalize_transform,imagenet_idx=True)
    print(f'Dataset has {len(train.classes)} classes. Sample classes: {train.classes[:5]}')
    
    # create dataloaders
    datasets = {"train": train,"val": val, "test": test}
    loaders = {}


    for name, dataset in datasets.items():
        print(f"Dataset size: {len(dataset)}")

        # create balanced batch sampler for training
        X = torch.stack([item[0] for item in dataset])
        targets = torch.tensor([item[1] for item in dataset])

        # create onehot vectors
        groups_onehot = torch.eye(200)[targets]

        # create a train dataset
        dataset_torch = torch.utils.data.TensorDataset(X, groups_onehot, targets)

        # create the balanced dataloader
        sampler = BalancedBatchSampler(
            group_onehot=groups_onehot, batch_size=batch_size, drop_last=True
        )
        loader_balanced = torch.utils.data.DataLoader(dataset_torch, batch_sampler=sampler, num_workers=10)
        loader_unbalanced = torch.utils.data.DataLoader(dataset_torch, batch_size=batch_size, shuffle=True, num_workers=10)

        # save the lodaers
        loaders[name] = loader_unbalanced
        loaders[name+"_balanced"] = loader_balanced

    # create fair dataloaders

    # test the models
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for name,dataloader in [ ('train',loaders['train_balanced']),('val',loaders['val_balanced']),('test',loaders['test_balanced']) ]:
        correct =0
        total = 0
        for (x,y, sens) in dataloader:

            print(x, y, sens)
            exit()

            pred = model(x)
            pred = pred.argmax(axis=1)
            total +=x.shape[0]
            correct += (y==pred).sum()

            print(x, y, pred)

            exit()
        accuracy = correct/total
        print(f'Accuracy for {name}: {accuracy}')


if __name__ == "__main__":
    train_tinyimagenet()