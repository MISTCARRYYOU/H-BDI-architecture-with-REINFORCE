
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 策略网络
# 一个简单的策略网络，面向的环境是 gym.CartPole
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64) # 网络第一层，输入是环境状态数，cartpole 状态为4维
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1) # 网络最后一层，输出是动作的维度，cartpole 仅2维，向左或向右，此处输出的是向左的概率
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # 前向计算
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x)) # sigmoid 函数保证概率在 [0,1]
        return x

    # 只是利用概率分布选择动作，先前向传播，再二值化的过程
    def select_action(self, state): # 为某状态选择一个动作
        with torch.no_grad():  # 这部分不被反向传播
            prob = self.forward(state)
            b = Bernoulli(prob) # pytorch 的伯努利分布
            action = b.sample()
            # prob: tensor([[0.5063]], device='cuda:0') action: tensor([[1.]], device='cuda:0')
            # prob: tensor([[0.5079]], device='cuda:0') action: tensor([[0.]], device='cuda:0')
            # print("prob:", prob, "action:", action)
        return action.item()

gamma = 0.99
optim = torch.optim.Adam  # 优化器
