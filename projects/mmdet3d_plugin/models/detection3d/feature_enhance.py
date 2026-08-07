import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # 计算 Q, K, V
        import pdb; pdb.set_trace()
        Q = self.query(x)  # (batch_size, seq_len, dim)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力得分
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)  # scaled dot-product
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用注意力权重
        out = torch.bmm(attention_weights, V)
        return out, attention_weights
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # 使用Sigmoid确保输出范围在[0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class FeatureEnhancer(nn.Module):
    def __init__(self, input_dim, noise_level=0.1):
        super(FeatureEnhancer, self).__init__()
        self.self_attention = SelfAttention(input_dim)
        self.noise_level = noise_level
        self.denoising_autoencoder = DenoisingAutoencoder(input_dim)  # 引入去噪自编码器
        self.fc = nn.Linear(input_dim, input_dim)  # 用于最终输出的线性层

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise

    def denoise(self, x):
        return self.denoising_autoencoder(x)


    def forward(self, x):
        # 添加噪声
        noisy_x = self.add_noise(x)

        # 使用自注意力机制
        enhanced_features, _ = self.self_attention.to(x.device)(noisy_x)

        # 去噪声
        denoised_features = self.denoise.to(x.device)(enhanced_features)

        # 最后的线性变换
        output = self.fc.to(x.device)(denoised_features)
        return output


# 测试 FeatureEnhancer
if __name__ == "__main__":
    # 假设的实例特征
    instance_features = torch.randn(6, 900, 256)  # (batch_size, seq_len, feature_dim)

    # 创建并运行特征增强器
    model = FeatureEnhancer(input_dim=256, noise_level=0.1)
    enhanced_instance_features = model(instance_features)
    import pdb; pdb.set_trace()
    print("原始特征形状:", instance_features.shape)
    print("增强后特征形状:", enhanced_instance_features.shape)
