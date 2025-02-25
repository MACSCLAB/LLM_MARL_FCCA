import torch.nn as nn
from .util import init, get_clones
import time

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal] #均匀初始化和正交初始化
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])  #1.414

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)  #初始化权重和偏置
        # nn.init.constant_(x, 0)用于将权重常数初始化，此处初始化为常数0

        #nn.Sequential接受一系列的子模块（layers）作为输入，并将其作为一个层,
        #在这里，第一个参数是init_过后的线性层，第二个参数是使用的激活函数
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size)) #这里是线性层 使用relu作为初始化方法？
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape=None, purpose='actor'):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        if purpose == 'actor':
            self.hidden_size = args.hidden_size
        else:
            self.hidden_size = args.hidden_size * 2

        if obs_shape is not None:
            obs_dim = obs_shape
        else:
            obs_dim = args.hidden_size

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x