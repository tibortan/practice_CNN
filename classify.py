import os
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


# 准备数据集，这里的数据是提前准备好的，一部分是脑部mri，一部分是膝部mri
class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = ['brain', 'knee']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.transform = transforms.Compose([
            # 用CenterCrop把里面有信息的框出来，分割算法后面再学
            transforms.CenterCrop((1024, 1024)),
            transforms.Resize((512, 512)),
            # 灰度图像，所以channel为1
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


# 输出长度
train_data_size = len(train_dataset)
test_data_size = len(val_dataset)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# batch这里设了几个感觉没太大区别，可能是数据集比较小，batch设不了太大
train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=True,
)

# 创建网络模型
class Yxlmod(nn.Module):
    def __init__(self):
        super(Yxlmod, self).__init__()
        self.model = nn.Sequential(
            # 大体是模拟了比较经典的VGG
            nn.Conv2d(1, 3, 5, 1, 2),
            # 很奇怪，加了激活函数反而训练不好了
            # nn.Sigmoid(),
            nn.MaxPool2d(4),
            nn.Conv2d(3, 16, 5, 1, 2),
            # nn.Sigmoid(),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 16, 5, 1, 2),
            # nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = self.model(x)
        return x


yxl = Yxlmod()
# 用我的1660s加速
yxl = yxl.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 这里的学习率，感觉和batch的设定互有影响
learning_rate = 0.01
optimizer = torch.optim.SGD(yxl.parameters(), lr=learning_rate)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤开始
    yxl.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = yxl(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

    # 测试步骤开始
    yxl.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in val_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = yxl(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    total_test_step = total_test_step + 1

    # 这看别人有保存，还没研究过保存的内容
    # torch.save(yxl, "yxl_{}.pth".format(i))
    # print("模型已保存")
