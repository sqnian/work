import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision

# 设置GPU(没有GPU则为CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)

# 导入数据库
import os
 
ROOT_FOLDER = 'data'
MNIST_FOLDER = ROOT_FOLDER + '/MNIST'
if not os.path.exists(MNIST_FOLDER) or not os.path.isdir(MNIST_FOLDER):
    print('开始下载数据集')
    # 下载训练集
    train_ds = torchvision.datasets.MNIST(ROOT_FOLDER, 
                                          train=True, 
                                          transform=torchvision.transforms.ToTensor(), # 将数据类型转化为Tensor
                                          download=True)
    # 下载测试集
    test_ds  = torchvision.datasets.MNIST(ROOT_FOLDER, 
                                          train=False, 
                                          transform=torchvision.transforms.ToTensor(), # 将数据类型转化为Tensor
                                          download=True)
else:
    print('数据集已下载 直接读取')
    # 读取已下载的训练集
    train_ds = torchvision.datasets.MNIST(ROOT_FOLDER, 
                                          train=True, 
                                          transform=torchvision.transforms.ToTensor(), # 将数据类型转化为Tensor
                                          download=False)
    # 读取已下载的测试集
    test_ds  = torchvision.datasets.MNIST(ROOT_FOLDER, 
                                          train=False, 
                                          transform=torchvision.transforms.ToTensor(), # 将数据类型转化为Tensor
                                          download=False)
    
#加载数据
batch_size = 32
# 从 train_ds 加载训练集
train_dl = torch.utils.data.DataLoader(train_ds, 
                                       batch_size=batch_size, 
                                       shuffle=True)
# 从 test_ds 加载测试集
test_dl  = torch.utils.data.DataLoader(test_ds, 
                                       batch_size=batch_size)
 
# 取一个批次查看数据格式
# 数据的shape为：[batch_size, channel, height, weight]
# 其中batch_size为自己设定，channel，height和weight分别是图片的通道数，高度和宽度。
imgs, labels = next(iter(train_dl))
print(imgs.shape)
# torch.Size([32, 1, 28, 28])  # 所有数据集中的图像都是28*28的灰度图

# 数据可视化
import numpy as np
 
# 指定图片大小，图像大小为20宽、5高的绘图(单位为英寸inch)
plt.figure('数据可视化', figsize=(20, 5)) 
for i, imgs in enumerate(imgs[:20]):
    # 维度缩减
    npimg = np.squeeze(imgs.numpy())
    # 将整个figure分成2行10列，绘制第i+1个子图。
    plt.subplot(2, 10, i+1)
    plt.imshow(npimg, cmap=plt.cm.binary)
    plt.axis('off')
plt.show()

# 构建CNN网络
import torch.nn.functional as F
 
num_classes = 10  # 图片的类别数
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
         # 特征提取网络
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 第一层卷积,卷积核大小为3*3
        self.pool1 = nn.MaxPool2d(2)                  # 设置池化层，池化核大小为2*2
        self.drop1 = nn.Dropout(p=0.15)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 第二层卷积,卷积核大小为3*3   
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(p=0.15)
        
        # 分类网络
        self.fc1 = nn.Linear(1600, 64)          
        self.fc2 = nn.Linear(64, num_classes)
    # 前向传播
    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))     
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
 
        x = torch.flatten(x, start_dim=1)
 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
       
        return x
    
# 加载并打印模型
from torchinfo import summary
# 将模型转移到GPU中（我们模型运行均在GPU中进行）
model = Model().to(device)
 
summary(model)

#设置超参数
loss_fn    = nn.CrossEntropyLoss() # 创建损失函数
learn_rate = 1e-2 # 学习率
opt        = torch.optim.SGD(model.parameters(),lr=learn_rate)

# 训练循环
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 训练集的大小，一共60000张图片
    num_batches = len(dataloader)   # 批次数目，1875（60000/32）
 
    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率
    
    for X, y in dataloader:  # 获取图片及其标签
        X, y = X.to(device), y.to(device)
        
        # 计算预测误差
        pred = model(X)          # 网络输出
        loss = loss_fn(pred, y)  # 计算网络输出和真实值之间的差距，targets为真实值，计算二者差值即为损失
        
        # 反向传播
        optimizer.zero_grad()  # grad属性归零
        loss.backward()        # 反向传播
        optimizer.step()       # 每一步自动更新
        
        # 记录acc与loss
        train_acc  += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()
            
    train_acc  /= size
    train_loss /= num_batches
 
    return train_acc, train_loss

def test (dataloader, model, loss_fn):
    size        = len(dataloader.dataset)  # 测试集的大小，一共10000张图片
    num_batches = len(dataloader)          # 批次数目，313（10000/32=312.5，向上取整）
    test_loss, test_acc = 0, 0
    
    # 当不进行训练时，停止梯度更新，节省计算内存消耗
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)
            
            # 计算loss
            target_pred = model(imgs)
            loss        = loss_fn(target_pred, target)
            
            test_loss += loss.item()
            test_acc  += (target_pred.argmax(1) == target).type(torch.float).sum().item()
 
    test_acc  /= size
    test_loss /= num_batches
 
    return test_acc, test_loss

epochs     = 50
train_loss = []
train_acc  = []
test_loss  = []
test_acc   = []
 
for epoch in range(epochs):
    model.train()
    epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_fn, opt)
    
    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_fn)
    
    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)
    
    template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%，Test_loss:{:.3f}')
    print(template.format(epoch+1, epoch_train_acc*100, epoch_train_loss, epoch_test_acc*100, epoch_test_loss))
print('Done')

import matplotlib.pyplot as plt
#隐藏警告
import warnings
warnings.filterwarnings("ignore")               #忽略警告信息
plt.rcParams['font.sans-serif']    = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
plt.rcParams['figure.dpi']         = 100        #分辨率
 
epochs_range = range(epochs)
 
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
 
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
 
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

"""
https://blog.csdn.net/ali1174/article/details/130294224

"""