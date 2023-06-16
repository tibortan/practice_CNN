import os
import sys

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# 准备数据集



class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = ['brain', 'knee']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.transform = transforms.Compose([
            transforms.CenterCrop((1024, 1024)),
            transforms.Resize((512, 512)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                sample = (img_path, self.class_to_idx[target_class])
                self.samples.append(sample)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)

train_dataset = MyDataset(root_dir='mri_out/train')
val_dataset = MyDataset(root_dir='mri_out/validation')

img, target = train_dataset[200]
plt.imshow(img.squeeze(), cmap='gray')
plt.show()
print(target)

# torch.set_printoptions(edgeitems=512, threshold=sys.maxsize)
# print(img)