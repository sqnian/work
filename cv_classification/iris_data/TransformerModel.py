import torch                        # 导入 PyTorch 库
from torch import nn                # 导入 PyTorch 的神经网络模块
from sklearn import datasets        # 导入 scikit-learn 库中的 dataset 模块
from sklearn.model_selection import train_test_split  # 从 scikit-learn 的 model_selection 模块导入 split 方法用于分割训练集和测试集
from sklearn.preprocessing import StandardScaler      # 从 scikit-learn 的 preprocessing 模块导入方法，用于数据缩放

print("# 检查GPU是否可用")
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("# 加载鸢尾花数据集")
# 加载鸢尾花数据集，这个数据集在机器学习中比较著名
iris = datasets.load_iris()
X = iris.data           # 对应输入变量或属性（features），含有4个属性：花萼长度、花萼宽度、花瓣长度 和 花瓣宽度
y = iris.target         # 对应目标变量（target），也就是类别标签，总共有3种分类

print("拆分训练集和测试")
# 把数据集按照80:20的比例来划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("数据缩放")
# 对训练集和测试集进行归一化处理，常用方法之一是StandardScaler
# Standardscaler是一种预处理技术，用于将数据缩放到均值为0、方差为1的标准正态分布
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("数据转tensor类型")
# 将训练集和测试集转换为PyTorch的张量对象并设置数据类型，加上to(device)可以运行在GPU上
X_train = torch.tensor(X_train).float().to(device)
y_train = torch.tensor(y_train).long().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_test = torch.tensor(y_test).long().to(device)

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()
        # 定义 Transformer 编码器，并指定输入维数和头数
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # 定义全连接层，将 Transformer 编码器的输出映射到分类空间
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # 在序列的第2个维度（也就是时间步或帧）上添加一维以适应 Transformer 的输入格式
        x = x.unsqueeze(1)
        # 将输入数据流经 Transformer 编码器进行特征提取
        x = self.encoder(x)
        # 通过压缩第2个维度将编码器的输出恢复到原来的形状
        x = x.squeeze(1)
        # 将编码器的输出传入全连接层，获得最终的输出结果
        x = self.fc(x)
        return x

print("创建模型")
# 初始化 Transformer 模型
model = TransformerModel(input_size=4, num_classes=3).to(device)

print("定义损失函数和优化器")
# 定义损失函数（交叉熵损失）和优化器（Adam）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("训练模型")
# 训练模型，对数据集进行多次迭代学习，更新模型的参数
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播计算输出结果
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播，更新梯度并优化模型参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印每10个epoch的loss值
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("测试模型")
# 测试模型的准确率
with torch.no_grad():
    # 对测试数据集进行预测，并与真实标签进行比较，获得预测
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Test Accuracy: {accuracy:.2f}')
