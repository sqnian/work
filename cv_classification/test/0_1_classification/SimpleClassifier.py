import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 创建一个简单的全连接神经网络模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 定义模型参数
input_dim = 10
hidden_dim = 5
output_dim = 2
model = SimpleClassifier(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 创建一些虚拟的训练数据
input_data = torch.randn(100, input_dim)
target = torch.randint(0, 2, (100,))


# 训练模型
for epoch in range(100):
    model.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# 测试模型
test_input = torch.randn(1, input_dim)
output = model(test_input)
predicted_class = torch.argmax(output).item()
print("Predicted class:", predicted_class)
