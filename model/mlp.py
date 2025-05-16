import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1, num_layers=8):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = F.relu(x)  #  使用ReLU激活函数
        x = self.layers[-1](x)  #  最后一层不使用激活函数
        return x

if __name__ == "__main__":
    #  ---  测试代码  ---
    batch_size = 2
    num_points = 64
    input_dim = 3  #  x, y, z

    #  创建一个随机输入
    test_input = torch.randn(batch_size, num_points, input_dim)

    #  创建一个MLP实例
    model = MLP(input_dim=input_dim, hidden_dim=128, num_layers=4)

    #  前向传播
    output = model(test_input)

    #  打印输出形状
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)  #  应该为 (batch_size, num_points, 1)