import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 创建一个简单的LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        output = self.fc(lstm_out[-1])
        return output

# 定义模型参数
input_dim = 10
hidden_dim = 5
output_dim = 2
model = LSTMModel(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 创建一些虚拟的训练数据
input_data = [torch.randn(1, input_dim) for _ in range(100)]
target = [np.random.choice([0, 1]) for _ in range(100)]
target = torch.LongTensor(target)

# 训练模型
for epoch in range(100):
    for i in range(len(input_data)):
        model.zero_grad()
        output = model(input_data[i])
        loss = criterion(output, target[i].unsqueeze(0))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# 测试模型
test_input = torch.randn(1, input_dim)
print("test_input :", test_input)
output = model(test_input)
print("output :", output)
predicted_class = torch.argmax(output).item()
print("Predicted class:", predicted_class)
