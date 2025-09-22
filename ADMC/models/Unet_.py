import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimestepEncoding(nn.Module):
    def __init__(self, d_embed, seq_len=1000):
        super(TimestepEncoding, self).__init__()
        self.d_model = d_embed
        pe = torch.zeros(seq_len, d_embed)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float()
            * (-math.log(10000.0) / d_embed)
        )
        pe[:, 0::2] = torch.sin(position * div_term)# 字嵌入维度为偶数时
        pe[:, 1::2] = torch.cos(position * div_term)# 字嵌入维度为奇数时
        self.register_buffer("pe", pe)

    def forward(self, t):
        x_pos = self.pe[t,:]# 变为x.size(1)长度，torch.Size([1, 4, 512])
        return x_pos#layer层会再加上位置信息


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.2, dim_feedforward = 512):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        # self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

    def forward(self, x, temb, key_padding_mask=None):
        # print(x.shape)#([320, 3, 256])
        # print(temb.shape)#([320, 256])
        x_ = x
        x = x + temb.unsqueeze(1)
        # 应用多头注意力机制，带有随机遮掩
        attn_output, attn_output_weights = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        # x = self.linear2(self.dropout(F.relu(self.linear1(attn_output))))
        # 归一化
        x = self.norm(self.dropout(attn_output)+x_)
        return x, attn_output_weights

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, temb):
        # print(x.shape, temb.shape)
        h = self.conv1(x)
        h = self.norm1(h)
        temb_proj = self.temb_proj(temb).unsqueeze(-1)
        # print(h.shape, temb_proj.shape)
        h = h + temb_proj
        h = self.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = h + temb_proj
        h = self.relu(h)

        return h


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128], d_model=96):
        super(UNet1D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(ResidualBlock(in_channels, feature, d_model))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(ResidualBlock(feature * 2, feature, d_model))

        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2, d_model)  # 中间块
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

        # 时间步编码
        self.pos_emb = TimestepEncoding(d_model)
        #可学习时间步
        self.weight_t = nn.parameter.Parameter(torch.randn(d_model))
        self.bias_t= nn.parameter.Parameter(torch.randn(d_model))
        # self.w = nn.Linear(self.h_dim * 2, self.h_dim)

        #增加多头注意力
        hidden_dim = 256
        num_heads = 4
        num_layers = 2
        self.layers = nn.ModuleList([MultiHeadAttentionLayer(hidden_dim, num_heads) for _ in range(num_layers)])
    def forward(self, x, t):

        # 可学习位置编码
        # if isinstance(t, int):
        #     t = torch.tensor([t], dtype=torch.float32, device=self.weight_t.device)
        # # print(t.unsqueeze(-1).shape)
        # temb = torch.cos(self.weight_t * t.unsqueeze(-1) + self.bias_t)
        # print(temb.shape)

        # 加上时间步编码
        temb = self.pos_emb(t)
        # print(temb.shape)#torch.Size([256])
        if len(temb.shape) == 1:
            temb = temb.unsqueeze(0).expand(x.size(0), -1)


        # 添加多层多头注意力
        # for layer in self.layers:
        #     x, attn_output_weights = layer(x, temb)

        skip_connections = []
        for layer in self.encoder:
            x = layer(x, temb)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, temb)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = self.resize_image(x, skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip, temb)
        out = self.final_conv(x)

        return out

    def resize_image(self, x, size):
        return nn.functional.interpolate(x, size=size, mode='linear', align_corners=True)


if __name__ == '__main__':
    # 示例用法
    batch_size = 8
    in_channels = 3
    out_channels = 3
    num_blocks = [32, 64, 128, 256]
    d_model = 96

    model = UNet1D(in_channels, out_channels, num_blocks, d_model)
    x = torch.randn(batch_size, in_channels, d_model)
    # timesteps = torch.randint(0, 1000, (batch_size,))
    timesteps = 10

    output = model(x, timesteps)
    print(output.shape)

