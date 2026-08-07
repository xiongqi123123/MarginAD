import torch
import torch.nn as nn

class ConsistencyLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(ConsistencyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embed_dim)  # 输出嵌入维度

    def forward(self, frames):
        """
        :param frames: 输入的帧特征，形状为 (batch_size, seq_length, embed_dim)
        :return: 输出的特征，形状为 (batch_size, 1, seq_length, embed_dim)
        """
        # 通过 LSTM 处理帧特征
        lstm_out, (hn, cn) = self.lstm(frames)  # lstm_out: (batch_size, seq_length, hidden_dim)

        # 通过全连接层得到预测
        output = self.fc(lstm_out)  # (batch_size, seq_length, embed_dim)

        # 调整形状以满足输出要求
        output = output.unsqueeze(1)  # (batch_size, 1, seq_length, embed_dim)
        
        return output

class NextTokenPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(NextTokenPredictor, self).__init__()
        # self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.lstm = ConsistencyLSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, embed_dim)  # 输出嵌入维度

    def forward(self, last_feat, last_score, cur_feat):
        # import pdb; pdb.set_trace()
        # 将历史特征根据分数加权
        weighted_last_feat = last_feat * last_score.sigmoid().unsqueeze(-1)  # (batch_size, 1, seq_length, embed_dim)

        # 将历史特征与当前特征结合
        combined_feat = cur_feat + weighted_last_feat  # (batch_size, 1, seq_length, embed_dim)

        # 调整维度以适应 LSTM 输入
        combined_feat = combined_feat.squeeze(1)  # (batch_size, seq_length, embed_dim)

        # 通过 LSTM 处理
        lstm_out = self.lstm.to(last_feat.device)(combined_feat)  # lstm_out: (batch_size, seq_length, hidden_dim)
        # import pdb; pdb.set_trace()
        # 通过全连接层得到预测
        # output = self.fc(lstm_out)  # (batch_size, seq_length, embed_dim)

        # 调整形状以满足输出要求
        # output = output.unsqueeze(1)  # (batch_size, 1, seq_length, embed_dim)
        
        return lstm_out
if __name__ == "__main__":
    # 超参数设置
    embed_dim = 256    # 特征维度
    hidden_dim = 128   # LSTM 隐藏层维度

    # 创建模型实例
    model = NextTokenPredictor(embed_dim, hidden_dim)

    # 示例输入
    batch_size = 6
    seq_length = 18

    last_feat = torch.randn(batch_size, 1, seq_length, embed_dim)  # 历史特征
    last_score = torch.randn(batch_size, 1, seq_length)             # 历史特征得分
    cur_feat = torch.randn(batch_size, 1, seq_length, embed_dim)    # 当前特征

    print("Input shape:", last_feat.shape,last_score.shape,cur_feat.shape)  # 应该是 (batch_size, 1, seq_length, embed_dim)

    # 前向传播
    output = model(last_feat, last_score, cur_feat)

    print("Output shape:", output.shape)  # 应该是 (batch_size, 1, seq_length, embed_dim)
