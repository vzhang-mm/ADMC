import copy
from typing import Optional
import math
import torch.nn.functional as F
from torch import nn, Tensor
import torch

class Transformer_encoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                dim_feedforward=1024, dropout=0.1,
                activation="relu", normalize_before=False,
                use_cls=False, use_pos=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,dropout, activation, normalize_before, use_cls)#1024是拼接后的维度
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self._reset_parameters()
        self.d_model = d_model
        self.use_cls = use_cls
        self.use_pos = use_pos

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # 可训练的CLS嵌入
            # 位置编码
            self.pos_emb = PositionalEncoding_(d_model)
        else:
            #时间步编码
            self.tim_emb = TimestepEncoding(d_model)
            # self.pos_emb = PositionalEncoding_(d_model)
            self.linear= nn.Sequential(
                nn.Linear(d_model, d_model*2),
                nn.ReLU(inplace=True),
                nn.Linear(d_model*2, d_model),
            )

        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers, encoder_norm)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, v, mask_v):#, query_embed, pos_embed
        if self.use_cls:
            # 将CLS嵌入复制到每个样本的批次中
            cls_tokens = self.cls_token.expand(v.size(0), -1, -1)  # 扩展CLS到批次大小
            v = torch.cat([cls_tokens, v], dim=1)  # 将CLS嵌入添加到序列前
            # 位置编码
            if self.use_pos:
                pos = self.pos_emb(v)
            else:
                pos = None
            # 更新掩码以包括CLS位置
            if mask_v is not None:
                cls_mask = torch.ones(v.size(0), 1).to(v.device)  # 创建CLS的掩码
                mask_v = torch.cat([cls_mask, mask_v], dim=1)  # 将CLS掩码添加到原有掩码前
            else:
                mask_v = torch.ones(v.size(0), v.size(1)).to(v.device)
            mask_v = mask_v.data.eq(0)#其中 mask 中的每个元素都与 0 进行比较。如果元素等于 0，则生成的布尔张量的对应位置为 True，否则为 False。
            v = self.encoder(v, src_key_padding_mask=mask_v, pos=pos)
            v = v[:, 0, :]  # 获取每个批次中CLS token的输出
            return v
        else:
            t = mask_v
            # 生成时间步编码
            tim_pos = self.tim_emb(t)
            #encoder
            out = self.encoder(v, pos=tim_pos)
            out = self.linear(out)
            return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, encoder_layer, num_layers, norm=None, use_cls=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers,)
        self.num_layers = num_layers
        self.norm = norm
        self.use_cls = use_cls

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, use_cls=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)#

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.use_cls = use_cls

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        if self.use_cls:
            q = k = self.with_pos_embed(src, pos)#
        else:
            if len(pos.shape) == 1:
                pos = pos.unsqueeze(0).expand(src.size(0), -1)
            #添加时间步编码
            pos = pos.unsqueeze(1).expand(src.size())
            q = k = self.with_pos_embed(src, pos)#
        # print('src_mask',src_mask)#None 防止序列中的某些部分关注到其他部分（如遮蔽自注意力在transformer的解码器中使用，以防止未来信息的泄露）,一般None
        # print('src_key_padding_mask', src_key_padding_mask.size())# 是用来指出哪些特定的序列元素（通常是填充的部分）在注意力机制中不应该被考虑的
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        '''
        Q 和 K 决定了序列中不同元素之间的相关性，即注意力权重。
        通过对 Q 和 K 添加位置编码，模型能够将输入序列中的位置信息融入到注意力权重的计算中。
        V 是用于加权求和的特征表示。因为它直接参与了输出的生成，理论上也可以添加位置编码。
        但通常位置编码在 Q 和 K 上已经足够了，因为这两者的作用是通过计算相关性来决定哪一部分信息应该被强调。
        '''
        q = k = self.with_pos_embed(src2, pos)#为什么只对Q,K添加位置编码？
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


############位置编码
class PositionalEncoding_(nn.Module):
    def __init__(self, d_embed, seq_len=5000):
        super(PositionalEncoding_, self).__init__()
        self.d_model = d_embed
        pe = torch.zeros(seq_len, d_embed)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float()
            * (-math.log(10000.0) / d_embed)
        )
        pe[:, 0::2] = torch.sin(position * div_term)# 字嵌入维度为偶数时
        pe[:, 1::2] = torch.cos(position * div_term)# 字嵌入维度为奇数时
        pe = pe.unsqueeze(0)## 在指定维度0上增加维度大小为1[3,4] -> [1,3,4]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)# sqrt() 方法返回数字x的平方根 适应多头注意力机制的计算，它说“这个数值的选取略微增加了内积值的标准差，从而降低了内积过大的风险，可以使词嵌入张量在使用多头注意力时更加稳定可靠”
        x_pos = self.pe[:, : x.size(1), :]# 变为x.size(1)长度，torch.Size([1, 4, 512])
        return x_pos#layer层会再加上位置信息


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




if __name__ == '__main__':
    model = Transformer_encoder(use_cls=True)
    batch_size = 32
    seq_len = 3
    d_model = 256

    # 沿指定维度堆叠张量
    src = torch.stack((torch.zeros((d_model)), torch.randn((d_model)),torch.randn((d_model))), dim=0).unsqueeze(0).repeat(8, 1, 1)
    print(src.shape)
    mask = torch.tensor([True,True,True]).unsqueeze(0).repeat(8, 1)#其中 True 表示需要遮掩的部分
    print(mask)

    # mask_v = mask.data.eq(0)#其中 mask 中的每个元素都与 0 进行比较。如果元素等于 0，则生成的布尔张量的对应位置为 True，否则为 False。
    # print('mask_v',mask_v)
    # 训练过程
    output = model(src, mask)
    print(output)

    # B, C, D = 1, 3, 256  # 批次大小、通道数和特征维度
    # data = torch.randn((B, C, D))
    #
    # # 定义模型
    # in_channels = C
    # blocks = [3,3,3,3]
    # model = Transformer_encoder()
    #
    # mask_ = torch.tensor([
    #     [True, False, True]
    # ], dtype=torch.bool)  # 掩码张量
    #
    # # 前向传播
    # output = model(data, mask_)
    # print(f'Output Shape: {output.shape}')

    # t = 1
    #
    # pe = time_encoding(d_embed = 256)
    # # 示例
    # x = torch.randn(8, 3, 256)  # 假设有8个样本，每个样本有3个通道，每个通道有256个特征
    # # pe = time_encoding[5]
    # print(x[0][0][:5])
    # out = x + pe[t]
    # print(pe[t][:5])
    # print(out[0][0][:5])
    # print(out.shape)

    # num_steps = 1000
    # d_embed = 256

    # 将位置编码与输入数据相加
    # features + pe
