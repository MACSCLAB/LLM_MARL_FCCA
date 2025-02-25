import torch
import torch.nn as nn
import time

class LSTMLayer(nn.Module):
    #TODO self_state_dim删掉
    def __init__(self, input_shape, lstm_hidden_dim, recurrent_N):
        super().__init__()
        self.input_dim = input_shape
        self.lstm_hidden_dim = lstm_hidden_dim
        #batch_first如果是Ture，那么batch_size在最前面；如果是False，那么sequence_length在最前面
        self.lstm = nn.LSTM(self.input_dim, lstm_hidden_dim, num_layers=recurrent_N, batch_first=True)  #初始化lstm

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)


    def forward(self, state): #输入张量格式(batch_size, sequence_length, input_dim)
        #(thread(batch),human_num,  input_size)
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        :param state: tensor of shape (batch_size, human n, length of a joint state)
        :return:
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        size = state.shape
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim) #为了记忆细胞的维度匹配，多加了一个维度
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        h0 = h0.to(device)
        c0 = c0.to(device)
        # state = self.feature_norm(state)
        output, (hn, cn) = self.lstm(state, (h0, c0))      #使用lstm,hn是前n个时间步的信息，cn是前面所有时间步的信息
        hn = hn.squeeze(0)#把前面多加的维度删掉
        return hn #[batch, lstm_hidden_dim]