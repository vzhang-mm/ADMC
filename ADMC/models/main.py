from models import transformer
from models import Diffusion, Unet_
# 构建模型
from torch import nn
from transformers import BertModel
import torch
import numpy as np
from opts import parse_opts
from models.lstm import LSTMEncoder
from models.textcnn import TextCNN
import pandas as pd


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

args = parse_opts()

use_Fusion = args.use_Fusion
use_text = args.use_text
use_vid = args.use_vid
use_wav = args.use_wav

use_FeatureExtraction=args.use_FeatureExtraction
use_DiffusionMlp = args.use_DiffusionMlp
use_MEIR = args.use_MEIR
use_MMIR = args.use_MMIR
use_zero = args.use_zero
use_modality = args.use_modality
use_full = args.use_full
N = args.N#固定MLP训练分类

print('是否是在全模型特征上添加噪声：',args.use_full)
def id_to_modality(use_modality):
    if use_modality == 0:
        modality = 'av'
    elif use_modality == 1:
        modality = 'ta'
    elif use_modality == 2:
        modality = 'tv'
    elif use_modality == 3:
        modality = 'a'
    elif use_modality == 4:
        modality = 'v'
    elif use_modality == 5:
        modality = 't'
    if use_MEIR:
        modality = 'tva'
    return modality

class Fusion_Attention(nn.Module):
    def __init__(self, input_dim=512):
        super(Fusion_Attention, self).__init__()
        self.multi_att = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        att_m = torch.softmax(self.multi_att(x), dim=1)
        #自定义权重
        # att_m =torch.tensor([1.0, 0.0, 0.0]).repeat(x.size(0), 1).unsqueeze(-1).to(device)
        x = torch.sum(x * att_m, dim=1)  # 沿着1维度求和
        return x
    

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.5):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 应用多头注意力机制
        attn_output, attn_output_weights = self.attn(x, x, x)
        # 残差连接和层归一化
        x = self.norm(x + self.dropout(attn_output))
        return x, attn_output_weights

class Fusion_MultiAttention(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_layers=1, num_heads=4):
        super(Fusion_MultiAttention, self).__init__()
        self.linear1= nn.Linear(text_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.linear2= nn.Linear(hidden_dim, text_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(text_dim)
        # 创建多个多头注意力层
        self.layers = nn.ModuleList([MultiHeadAttentionLayer(text_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x, attn_output_weights = layer(x)
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = self.norm(output)
        # 均值聚合
        output = output.mean(dim=1)##这里肯定不行
        return output


class Fusion_Cat(nn.Module):
    def __init__(self,):
        super(Fusion_Cat, self).__init__()

    def forward(self, tva_x):
        return tva_x.view(tva_x.size(0),-1)
    
def compute_metrics(c_loss_fn, output, labels):

    # 使用交叉熵损失函数
    c_loss = c_loss_fn(output, labels)
    # 计算准确度
    preds = output.argmax(dim=1)
    return c_loss, preds

###################################################################################
# 六种不同的缺失状态
states = [
    [1, 0, 0],  # text missing
    [0, 1, 0],  # vid missing
    [0, 0, 1],  # wav missing
    [1, 1, 0],  # text and vid missing
    [1, 0, 1],  # text and wav missing
    [0, 1, 1],  # vid and wav missing
]


class feature_extraction(nn.Module):
    warning = True
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.input_dim = 256
        self.output_dim = 256

        if args.dataset == 'IEMOCAP':
            self.num_classes = 4
            self.netA = LSTMEncoder(130, self.input_dim, embd_method='maxpool')
            self.netL = TextCNN(1024, self.input_dim)
            self.netV = LSTMEncoder(342, self.input_dim, embd_method='maxpool')
        elif args.dataset == 'MIntRec':
            self.num_classes = 20
            self.netA = LSTMEncoder(768, self.input_dim, embd_method='maxpool')
            self.netL = TextCNN(768, self.input_dim)
            self.netV = LSTMEncoder(256, self.input_dim, embd_method='maxpool')

            # self.netA = nn.Sequential(
            #     nn.Linear(768, self.input_dim),
            #     transformer.Transformer_encoder(d_model=self.input_dim, dim_feedforward=self.input_dim * 4,
            #                                     num_encoder_layers=2, use_cls=True))
            # self.netL = nn.Sequential(
            #     nn.Linear(768, self.input_dim),
            #     transformer.Transformer_encoder(d_model=self.input_dim, dim_feedforward=self.input_dim * 4,
            #                                     num_encoder_layers=2, use_cls=True))
            #
            # self.netV = transformer.Transformer_encoder(d_model=self.input_dim, dim_feedforward=self.input_dim * 4,
            #                                     num_encoder_layers=2, use_cls=True)

        else:
            self.num_classes = 20
            self.Text_model = BertEmbedding()
            self.Vid_model = transformer.Transformer_encoder(d_model=self.input_dim, dim_feedforward=self.input_dim * 4,
                                                             num_encoder_layers=2, use_cls=True)
            self.Wav_model = nn.Sequential(
                nn.Linear(768, self.input_dim),
                transformer.Transformer_encoder(d_model=self.input_dim, dim_feedforward=self.input_dim * 4,
                                                num_encoder_layers=2, use_cls=True))
        self.data_list = []

    def forward(self, text, vid, wav, labels):
        v, mask_v = vid['vids'], vid['vid_masks']
        a, mask_a = wav['wavs'], wav['wav_masks']

        if args.dataset == 'IEMOCAP':
            t, mask_t = text['texts'], text['text_masks']  # IEMOCAP
            vid_x = self.netV(v)
            wav_x = self.netA(a)
            text_x = self.netL(t)
        elif args.dataset == 'MIntRec':
            t, mask_t = text['texts'], text['text_masks']  # MIntRec
            # vid_x = self.netV(v, mask_v)
            # wav_x = self.netA[1](self.netA[0](a), mask_a)
            # text_x = self.netL[1](self.netL[0](t), mask_t)

            vid_x = self.netV(v)
            wav_x = self.netA(a)
            text_x = self.netL(t)
        else:
            t = text['texts']  # MIntRec
            vid_x = self.Vid_model(v, mask_v)
            wav_x = self.Wav_model[1](self.Wav_model[0](a), mask_a)
            text_x = self.Text_model(t)
        # text_x
        combined_features = torch.cat((text_x,vid_x,wav_x,labels.view(labels.size(0), 1)),dim = 1)
        self.data_list.append(combined_features.cpu().detach().numpy())

        ######训练特征提取网络################
        if use_FeatureExtraction:
            if use_Fusion:
                # 混合训练特征提取
                if classifier.warning:
                    print('利用tva训练模型')
                    classifier.warning = False
                tva_data = torch.stack((text_x, vid_x, wav_x), dim=1)  # t,v,a
                return tva_data
            else:
                # 单独训练特征提取
                if use_text:
                    if classifier.warning:
                        print('利用text训练模型')
                        classifier.warning = False
                    output = text_x
                elif use_vid:
                    if classifier.warning:
                        print('利用vid训练模型')
                        classifier.warning = False
                    output = vid_x
                elif use_wav:
                    if classifier.warning:
                        print('利用wav训练模型')
                        classifier.warning = False
                    output = wav_x
                return output
        else:
            return text_x,vid_x,wav_x

    def save_data_epoch(self, epoch):
        # 将所有 data_0 合并成一个数组
        data_0_epoch = np.concatenate(self.data_list, axis=0)
        print('data_0_epoch', len(data_0_epoch))
        df = pd.DataFrame(data_0_epoch)
        file_path = f'data_0_epoch_{epoch}.csv'
        df.to_csv(file_path, index=False)
        print(f'Saved data_0 for epoch {epoch} to CSV.')
        # 清空列表，为下一个 epoch 做准备
        self.data_list.clear()

class CDMC(nn.Module):##训练MLP
    warning = True
    def __init__(self):
        super(CDMC, self).__init__()
        self.output_dim = 256

        # 初始化Diffusion
        # self.diffusion = Diffusion.DiffusionModel(beta_start=3e-4, beta_end=6e-2, num_steps=500)  # 100，500，1000
        self.diffusion = Diffusion.DiffusionModel(beta_start=2e-4, beta_end=4e-2, num_steps=1000)  # 100，500，1000
        self.mlp_model = Unet_.UNet1D(in_channels=3, out_channels=3, d_model=self.output_dim)  # 3f
        # self.mlp_model = transformer.Transformer_encoder(d_model=self.output_dim, dim_feedforward=self.output_dim*4, num_encoder_layers=4, nhead=8)

        self.d_loss = nn.MSELoss(reduction='none')

    def forward(self, text_x, vid_x, wav_x):
        # 数据重建
        tva_data = torch.stack((text_x, vid_x, wav_x), dim=1).to(device)
        B, C, D = tva_data.size()
        # 初始化形状为 (6, B, 3, D) 的张量来存储所有缺失状态数据
        all_missing_states = torch.zeros((6, B, C, D), device=tva_data.device)
        # 初始化形状为 (6, B, 3) 的布尔张量来存储所有缺失状态掩码
        all_missing_masks = torch.zeros((6, B, C), dtype=torch.bool, device=tva_data.device)
        if use_zero:
            noise = torch.zeros((B, D), device=tva_data.device)
        else:
            noise = torch.randn((B, D), device=tva_data.device)
        for idx, state in enumerate(states):
            # 复制数据
            data = tva_data.clone()
            mask = torch.zeros((B, C), dtype=torch.bool, device=tva_data.device)
            for i in range(C):
                if state[i] == 1:
                    data[:, i, :] = noise  # 将缺失模态的特征值设为随机噪声
                    mask[:, i] = True  # 生成缺失掩码,其中 True 表示需要遮掩的部分
            all_missing_states[idx] = data
            all_missing_masks[idx] = mask

        if use_DiffusionMlp:
            all_d_loss = 0.0
            d_loss_class = []
            if classifier.warning:
                print('#####################################训练MLP loss')
                classifier.warning = False

            for i in range(all_missing_states.size(0)):
                data_ = all_missing_states[i].to(device)
                mask_ = all_missing_masks[i].to(device).unsqueeze(-1).expand(data_.size())
                # 目标数据（tva_data）,直接在所有模态上加噪
                ti = torch.randint(0, self.diffusion.num_steps, (tva_data.shape[0],)).long().to(device)
                noise = torch.randn_like(tva_data).to(device)
                noise_features = self.diffusion.q_sample(tva_data, ti, noise).to(device)
                if use_full:  # 老师方法在全局上加噪
                    pass
                else:  # 保持未缺失模态特征不变
                    noise_features = torch.where(mask_, noise_features,tva_data)  # mask为True的位置用在指定位置加噪后的noise_features代替
                pred_noise = self.mlp_model(noise_features, ti)
                d_loss = self.d_loss(pred_noise, noise)
                if use_full:
                    d_loss = d_loss.mean()
                    d_loss_class.append([d_loss.item()])
                else:
                    # 只计算mask为True位置的
                    masked_loss = d_loss[mask_]
                    d_loss = masked_loss.mean()
                    d_loss_class.append([d_loss.item()])
                all_d_loss = all_d_loss + d_loss

            return d_loss_class, None, all_d_loss  # 返回每个类别的loss

        else:  ##\MMIR\MEIR\0
            if use_MEIR:
                if classifier.warning:
                    print("#####################################use_MEIR")
                    classifier.warning = False
                if use_full:  # 老师的方法
                    fake_features = torch.zeros((B, C, 2*D), device=tva_data.device)
                    for i in [0,1,2]:
                        data_ = all_missing_states[i].to(device)
                        mask = all_missing_masks[i].to(device)  # mask为True的部分表示缺失
                        noise = torch.randn_like(data_).to(device)
                        # 前向加N时间步噪声
                        ti = torch.full((data_.size(0),), N, device=device).long()
                        data_ = self.diffusion.q_sample(data_, ti, noise)
                        # # 随机噪声整体降噪到num_steps-N步
                        noise_N = self.diffusion.generate_samples(self.mlp_model, noise,num_steps=self.diffusion.num_steps - N)
                        # # 两者拼接，作为去噪的输入
                        expanded_mask = mask.unsqueeze(2).expand(-1, -1, data_.size(-1))
                        result = torch.where(expanded_mask, noise_N, data_)  # mask为True的位置是noise_N
                        fake_features_i = self.diffusion.generate_samples(self.mlp_model, result, num_steps=N, mask=None)
                        fake_features[:,i,:] = torch.cat((fake_features_i[:, i, :], tva_data[:, i, :]), dim=-1)

            elif use_MMIR:  ###指定训练单个缺失模态
                if classifier.warning:
                    print("#####################################use_MMIR")
                    classifier.warning = False

                data = all_missing_states[use_modality].to(device)
                mask = all_missing_masks[use_modality].to(device)  # mask为True的部分表示缺失
                if use_full:  # 老师的方法
                    # 对data整体加噪到N步
                    # data_ = tva_data#验证
                    data_ = data
                    noise = torch.randn_like(data_).to(device)
                    # 前向加N时间步噪声
                    ti = torch.full((data.size(0),), N, device=device).long()
                    data_ = self.diffusion.q_sample(data_, ti, noise)
                    # fake_features = data_##只加噪

                    # 反向从N步降噪到0
                    # fake_features = self.diffusion.generate_samples(self.mlp_model,  data_, num_steps=N)#验证

                    # # 随机噪声整体降噪到num_steps-N步
                    noise_N = self.diffusion.generate_samples(self.mlp_model, noise, num_steps=self.diffusion.num_steps-N)
                    # # 两者拼接，作为去噪的输入
                    expanded_mask = mask.unsqueeze(2).expand(-1, -1, data_.size(-1))
                    result = torch.where(expanded_mask, noise_N, data_)  # mask为True的位置是noise_N
                    #从N降到0
                    fake_features = self.diffusion.generate_samples(self.mlp_model, result, num_steps=N, mask=None)
                    # 还要利用mask将原data与fake_features拼接
                    fake_features = torch.where(expanded_mask, fake_features, data)  # mask为True的位置用生成的fake_features代替
                else:
                    if args.use_DDIM:
                        #DDIM
                        print('*****************使用DDIM')
                        fake_features = self.diffusion.generate_samples(self.mlp_model, data,num_steps=self.diffusion.num_inference_steps, mask=mask)
                    else:
                        # 将噪声填充的缺失模态特征作结合已有模态特征（条件）作为输入,从T开始还原
                        fake_features = self.diffusion.generate_samples(self.mlp_model, data,num_steps=self.diffusion.num_steps, mask=mask)
            elif use_zero:
                # 使用0补全
                if classifier.warning:
                    print("#####################################use_zero")
                    classifier.warning = False
                fake_features = all_missing_states[use_modality].to(device)

            return fake_features



class MF(nn.Module):
    warning = True
    def __init__(self, class_weights):
        super(MF, self).__init__()
        if args.use_MEIR:
            self.output_dim = 256*2
        else:
            self.output_dim = 256
        if args.dataset == 'IEMOCAP':
            self.num_classes = 4
        else:
            self.num_classes = 20

        # #特征融合
        self.Fusion_model = Fusion_MultiAttention(text_dim=self.output_dim, hidden_dim=self.output_dim * 4,
                                                      num_layers=2, num_heads=4)
        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(self.output_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.num_classes),
            nn.ReLU(),
        )

        # 使用加权损失函数
        self.c_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, fake_features, labels):
        if use_FeatureExtraction:
            if use_Fusion:
                tva_data = fake_features
                output = self.fc(self.Fusion_model(tva_data))
                c_loss, preds = compute_metrics(self.c_loss, output, labels)
                return preds, c_loss, None
            else:
                x_data = fake_features
                output = self.fc(x_data)
                c_loss, preds = compute_metrics(self.c_loss, output, labels)
                return preds, c_loss, None
        else:
            output = self.fc(self.Fusion_model(fake_features))
            c_loss, preds = compute_metrics(self.c_loss, output, labels)  # 6种表现形式，对应的都是一种label
            return preds, c_loss, None  # 返回每个类别的loss


class classifier(nn.Module):
    warning = True
    def __init__(self, class_weights):
        super(classifier, self).__init__()

        self.feature_extraction_model = feature_extraction()
        self.CDMC_model = CDMC()

        self.MF_model = MF(class_weights)
        self.fake_features_list = []
    def forward(self, text, vid, wav, labels):
        if use_FeatureExtraction:
            if use_Fusion:
                tva_data = self.feature_extraction_model(text, vid, wav,labels)
                return self.MF_model(tva_data,labels)
            else:
                x_data = self.feature_extraction_model(text, vid, wav, labels)
                return self.MF_model(x_data, labels)
        else:
            #单独特征提取器
            text_x, vid_x, wav_x = self.feature_extraction_model(text, vid, wav, labels)
            ##增加融合特征提取器
            if use_DiffusionMlp:
                return self.CDMC_model(text_x, vid_x, wav_x)
            else:
                fake_features = self.CDMC_model(text_x, vid_x, wav_x)

                # 拼接 fake_features_reshaped 和 labels_expanded
                combined_features = torch.cat((fake_features.view(labels.size(0), -1), labels.view(labels.size(0), 1)), dim=1)
                self.fake_features_list.append(combined_features.cpu().detach().numpy())

                return self.MF_model(fake_features,labels)

    def save_fake_features_epoch(self, epoch):
        modality = id_to_modality(args.use_modality)
        # 将所有 data_0 合并成一个数组
        data_0_epoch = np.concatenate(self.fake_features_list, axis=0)
        print('fake_features_epoch', len(data_0_epoch))
        df = pd.DataFrame(data_0_epoch)
        file_path = f'fake_features_epoch_{modality}_{epoch}.csv'
        df.to_csv(file_path, index=False)
        print(f'Saved fake_features for epoch {epoch} to CSV.')
        # 清空列表，为下一个 epoch 做准备
        self.fake_features_list.clear()


class BertEmbedding(nn.Module):
    def __init__(self, dropout=0.2):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('./bert_base_uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        mask = x['attention_mask'].squeeze(1)
        input_id = x['input_ids'].squeeze(1)
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask,return_dict=False)  # token_type_ids = token_type_ids,
        dropout_output = self.dropout(pooled_output)
        output = self.relu(self.linear(dropout_output))  # 768->256
        return output


if __name__ == '__main__':
    pass
    for i in range(6):
        noise = torch.randn(3, 32)
        print(noise [0,:5])