import torch
from torch.utils.data import Dataset, Subset,DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop


class Dataset_diffusion_models():
    def __init__(self,fashion=True, batch_size=32):
        super().__init__()

        transform = Compose([
            ToTensor(),
            Lambda(lambda x: (x - 0.5) * 2)]
        )

        self.batch_size = batch_size

        ds_fn = FashionMNIST if fashion else MNIST

        self.train_dataset = ds_fn("./datasets", download=True, transform=transform, train=True)
        self.val_dataset = ds_fn("./datasets", download=True,transform=transform, train=False)


        self.train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False,batch_size=1)

    
    def data_loader(self):
        return self.train_loader, self.val_loader 








