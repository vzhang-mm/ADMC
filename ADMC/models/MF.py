import pandas as pd
from torch.utils.data import Dataset, DataLoader,ConcatDataset
import torch
from torch import nn
from torch.optim import AdamW,Adam
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from models import transformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

def compute_metrics(preds, labels):
    # 转换为张量
    preds = torch.stack(preds, dim=0)
    labels = torch.stack(labels, dim=0)

    # 转换为 numpy 数组
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 总体准确率
    acc = accuracy_score(labels_np, preds_np)

    #WA
    unique_labels = np.unique(labels_np)
    # 按类别计算准确率和权重
    accuracies = []
    weights = []
    for label in unique_labels:
        mask = (labels_np == label)
        acc = accuracy_score(labels_np[mask], preds_np[mask])  # 每个类别的准确率
        weight = np.sum(mask)  # 每个类别的样本数量
        accuracies.append(acc)
        weights.append(weight)

    # 计算加权平均
    WA = np.sum(np.array(accuracies) * np.array(weights)) / np.sum(weights)

    # Precision, Recall, F1 (macro 平均)
    precision = precision_score(labels_np, preds_np, average='weighted', zero_division=0)#'macro'
    recall = recall_score(labels_np, preds_np, average='macro', zero_division=0)  # UA
    f1 = f1_score(labels_np, preds_np, average ='weighted', zero_division=0)################# 'weighted'就是WAF

    return round(WA, 4), round(precision, 4), round(recall, 4), round(f1, 4)


class CustomDataset(Dataset):
    def __init__(self, csv_file_list, part='train'):
        if use_MIntRec:
            train_size = 1334
            val_size = 443
            test_size = 443
        else:
            train_size = 4446
            val_size = 557
            test_size = 528

        self.data = pd.read_csv(csv_file_list)
        if part == 'train':
            self.data = self.data.iloc[:train_size, :]
        elif part == 'val':
            self.data = self.data.iloc[train_size:train_size+val_size, :]
            # self.data = self.data.iloc[train_size:, :]
        else:
            self.data = self.data.iloc[train_size+val_size:, :]
            # self.data = self.data.iloc[train_size:, :]

        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data.iloc[:, -1].values.astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return sample

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.2):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # 应用多头注意力机制，带有随机遮掩
        attn_output, attn_output_weights = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        # 残差连接和层归一化
        x = self.norm(x + self.dropout(attn_output))
        return x, attn_output_weights


class Fusion_MultiAttention_(nn.Module):

    def __init__(self, text_dim, hidden_dim, num_layers=1, num_heads=4):
        super(Fusion_MultiAttention_, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vid_proj = nn.Linear(text_dim, hidden_dim)
        self.wav_proj = nn.Linear(text_dim, hidden_dim)

        # 创建多个多头注意力层
        self.layers = nn.ModuleList([MultiHeadAttentionLayer(hidden_dim, num_heads) for _ in range(num_layers)])

        self.linear= nn.Sequential(
            nn.Linear(hidden_dim, text_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, text_x,vid_x,wav_x,):
        text_x = self.text_proj(text_x)
        vid_x = self.vid_proj(vid_x)
        wav_x = self.wav_proj(wav_x)

        # 堆叠特征 (B, 3, hidden_dim)
        output = torch.stack((text_x, vid_x, wav_x), dim=1)  # text_x,
        # 通过多层多头注意力
        for layer in self.layers:
            output, attn_output_weights = layer(output)
        # 均值聚合
        output = output.mean(dim=1)
        output = self.linear(output)

        return output


class Fusion_MultiAttention(nn.Module):
    def __init__(self, text_dim, hidden_dim, num_layers=1, num_heads=4):
        super(Fusion_MultiAttention, self).__init__()

        self.linear1= nn.Linear(text_dim, hidden_dim)
        self.dropout =  nn.Dropout(0.5)
        self.linear2= nn.Linear(hidden_dim, text_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(text_dim)
        # 创建多个多头注意力层
        self.layers = nn.ModuleList([MultiHeadAttentionLayer(text_dim, num_heads) for _ in range(num_layers)])

        # self.encoder = transformer.Transformer_encoder(d_model=text_dim, dim_feedforward=hidden_dim,
        #                                                num_encoder_layers=num_layers, nhead=num_heads, use_cls=True, use_pos=False)

    def forward(self, text_x,vid_x,wav_x, apply_mask=False):
        # 生成随机遮掩掩码 (B, 3) 用于多模态输入 (text, vid, wav)
        if apply_mask:
            key_padding_mask = torch.rand(text_x.size(0), 3, device='cuda') < 0.2  # 50% 的概率随机遮掩一个模态
            # 确保每个样本至少有一个模态未被遮掩
            while key_padding_mask.sum(dim=1).eq(3).any():
                # 如果有样本的所有模态都被遮掩了，重新生成这个样本的掩码
                key_padding_mask = torch.where(
                    key_padding_mask.sum(dim=1, keepdim=True).eq(3),
                    torch.rand(text_x.size(0), 3, device='cuda') < 0.2,
                    key_padding_mask
                )
            # 将布尔张量转换为 0 和 1 的整数张量，然后反转 0 和 1
            # key_padding_mask = (1 - key_padding_mask.to(torch.int))
        else:
            key_padding_mask = None  # 不应用遮掩
        # key_padding_mask = None  # 不应用遮掩
        # 堆叠特征 (B, 3, hidden_dim)
        x = torch.stack((text_x, vid_x, wav_x), dim=1)

        for layer in self.layers:
            x, attn_output_weights = layer(x, key_padding_mask=key_padding_mask)
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = self.norm(output)
        # 均值聚合
        output = output.mean(dim=1)##这里肯定不行
        # output = output[:, 1, :]  # 仅使用 text 模态的特征

        # output = self.encoder(x, key_padding_mask)
        return output

class Fusion_Cat(nn.Module):
    def __init__(self,output_dim):
        super(Fusion_Cat, self).__init__()
        self.output_dim = output_dim

        self.linear = nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.ReLU(),
        )

    def forward(self, text_x,vid_x,wav_x):
        text_x = self.linear(text_x)
        vid_x = self.linear(vid_x )
        wav_x = self.linear(wav_x)

        x = torch.cat((text_x, vid_x, wav_x), dim=-1)

        return x


class Classifier(nn.Module):
    def __init__(self, modality=None):
        super(Classifier, self).__init__()
        if use_MMER:
            self.input_dim = 256*2
        else:
            self.input_dim = 256
        self.output_dim = self.input_dim

        if use_MIntRec:
            self.num_classes = 20
        else:
            self.num_classes = 4########4

        self.fusion = Fusion_MultiAttention_(text_dim=self.input_dim, hidden_dim=self.output_dim*4, num_layers=2, num_heads=4)

        self.modality = modality
        if mlp_work:
            # 分类层
            self.fc = nn.Sequential(
                nn.Linear(self.output_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes),
                nn.ReLU(),
            )
        else:
            # 分类层
            self.fc = nn.Sequential(
                nn.Linear(self.output_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, self.num_classes),
                nn.ReLU(),
            )

            # self.fc = nn.Sequential(
            #     nn.Linear(768, 512),
            #     nn.ReLU(),
            #     nn.Linear(512, 256),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(256),
            #     nn.Linear(256, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, self.num_classes),
            #     nn.ReLU(),
            # )

    def forward(self, x):
        if use_MMER:
            text_x = x[:, 0:512]
            vid_x = x[:, 512:1024]
            wav_x = x[:, 1024:]
        else:
            text_x = x[:, 0:256]
            vid_x = x[:, 256:512]
            wav_x = x[:, 512:]

        if use_ITFN:
            if self.modality == 'av':
                text_x = torch.zeros_like(text_x).to('cuda')
            elif self.modality == 'ta':
                vid_x = torch.zeros_like(vid_x).to('cuda')
            elif self.modality == 'tv':
                wav_x = torch.zeros_like(wav_x).to('cuda')
            elif self.modality == 'a':
                text_x = vid_x = torch.zeros_like(text_x).to('cuda')
            elif self.modality == 'v':
                text_x = wav_x = torch.zeros_like(wav_x).to('cuda')
            elif self.modality == 't':
                wav_x = vid_x = torch.zeros_like(wav_x).to('cuda')

        if mlp_work:
            if self.modality == 'av':
                x = text_x
            elif self.modality == 'ta':
                x = vid_x
            elif self.modality == 'tv':
                x = wav_x
            elif self.modality == 'a':
                x = text_x
                # x = vid_x
            elif self.modality == 'v':
                x = text_x
                # x = wav_x
            elif self.modality == 't':
                x = wav_x
                # x = vid_x

        else:
            x = self.fusion(text_x,vid_x,wav_x)
            # x = torch.cat((text_x,vid_x,wav_x),dim=-1)
        out = self.fc(x)
        return out


# 测试模型
def evaluate(model, test_dataloader):
    model.eval()
    test_total_preds = []
    test_total_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            features = batch['features'].to('cuda')
            test_labels = batch['label'].to('cuda')
            output = model(features)
            test_preds = output.argmax(dim=1)
            test_total_preds.extend(test_preds)
            test_total_labels.extend(test_labels)
        test_acc, test_precision, test_recall, test_f1 = compute_metrics(test_total_preds, test_total_labels)

    if test_acc > 0.6:
        print(
            f"***********************\n"
            f"test WA: {test_acc}\n"
            f"test UA: {test_recall}\n"
            f"test F1: {test_f1}\n"
        )

def train(model, train_dataloader, val_dataloader, test_dataloader, class_weights):

    criterion = nn.CrossEntropyLoss(class_weights)#weight=class_weights
    # optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # 设置学习率调度器
    num_training_steps = epochs * len(train_dataloader)# 实际训练步骤数
    num_warmup_steps = int(0.1 * num_training_steps)  # 设置为总训练步骤数的 10%,30epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    best_WA = 0.0
    best_UA = 0.0
    best_F1 = 0.0
    best_precision = 0.0
    best_WAF = 0.0
    for epoch_num in range(epochs):
        model.train()
        total_loss_train = 0.0
        total_preds = []
        total_labels = []

        all_total_acc_val = []
        total_loss_val = 0.0
        val_total_preds = []
        val_total_labels = []

        for batch in train_dataloader:# six，av,ta,tv,a,v,t
            optimizer.zero_grad()
            features = batch['features'].to('cuda')
            labels = batch['label'].to('cuda')

            output = model(features)
            loss = criterion(output, labels)
            preds = output.argmax(dim=1)
            total_loss_train += loss.item()

            total_preds.extend(preds)
            total_labels.extend(labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

        train_acc, train_precision, train_recall, train_f1 = compute_metrics(total_preds, total_labels)

        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                features = batch['features'].to('cuda')
                val_labels = batch['label'].to('cuda')

                output = model(features)
                val_preds = output.argmax(dim=1)
                loss = criterion(output, val_labels)
                total_loss_val += loss.item()

                val_total_preds.extend(val_preds)
                val_total_labels.extend(val_labels)

        val_acc, val_precision, val_recall, val_f1 = compute_metrics(val_total_preds, val_total_labels)
        all_total_acc_val.append([val_acc, val_precision, val_recall, val_f1])

        if val_acc > best_WA:
            best_WA = val_acc#acc
            best_F1 = val_f1
            best_precision = val_precision
            best_UA  = val_recall#

            print(f"Epoch {epoch_num+1}/{epochs}\n")
            evaluate(model, test_dataloader)
            print(
                f"######################################\n"
                f"第{i}轮交叉验证"
                f"Train Acc: {train_acc}\n"
                f"Train recall: {train_acc}\n"
                f"Val WA/acc: {val_acc}\n"
                f"Val precision: {val_precision}\n"
                f"Val UA/recall : {val_recall}\n"
                f"Val WAF/f1: {val_f1}\n"
                f"######################################\n"
            )
        else:
            pass
            # if (epoch_num + 1) % 20 == 0:
            #     print(f"Epoch {epoch_num+1}/{epochs}\n")
            #     print(
            #         f"Train Acc: {train_acc}\n"
            #         f"Train recall: {train_acc}\n"
            #         f"Val WA: {val_acc}\n"
            #         # f"Val UA: {val_recall}\n"
            #         f"Val F1: {val_f1}\n"
            #     )

    return best_WA, best_precision, best_UA, best_F1


    # labels = [sample['label'].item() for sample in train_dataset]
    # # 计算每个类别的权重
    # class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    # class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda')
    # print(class_weights)

def load_data(modality):
    if use_ITFN:
        csv_file = f'D:/Desktop/访谈系统论文/模态缺失问题/IE_data/use_ITFN/data_0_epoch_0.csv'
    else:
        #use_ADN:
        # csv_file = f'D:/Desktop/访谈系统论文/模态缺失问题/IE_data/use_ADN/fake_features_epoch_{modality}_0.csv'

        # MMIR
        # csv_file = f'D:/Desktop/访谈系统论文/模态缺失问题/MI_data/50/fake_features_epoch_{modality}_0.csv'
        # csv_file = f'D:/Desktop/访谈系统论文/模态缺失问题/IE_data/50/fake_features_epoch_{modality}_0.csv'
        print('fff')
        csv_file = f'D:/Desktop/ADMC_02/IE_50_notfull/fake_features_epoch_{modality}_0.csv'

        #unet
        # csv_file = f'D:/Desktop/访谈系统论文/模态缺失问题/IE_data/unet/fake_features_epoch_{modality}_0.csv'
        #补充实验2分类
        # csv_file = f'D:/Desktop/访谈系统论文/模态缺失问题/IE_data/output_files/data_missing_{modality}.csv'
        #补充实验4分类
        # csv_file = f'D:/Desktop/访谈系统论文/模态缺失问题/IE_data/four/data_missing_{modality}.csv'

    # 创建数据集
    train_dataset = CustomDataset(csv_file, part='train')
    val_dataset = CustomDataset(csv_file, part='val')
    test_dataset = CustomDataset(csv_file, part='test')

    train_dataloader = DataLoader(train_dataset, batch_size=320, shuffle=False)
    # 创建并保存验证集和测试集的DataLoader
    val_dataloader = DataLoader(val_dataset, batch_size=320, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=320, shuffle=False)

    print("训练集数量：",len(train_dataloader))

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':

    ##########3个只能选择一个模型
    use_MMER = False
    mlp_work = False#检查生成的特征是否有效
    use_ITFN = False#固定特征提取网络权重，其他用0补全
    use_MIntRec = False

    if mlp_work:
        epochs = 4000
        learning_rate = 1e-3
    else:
        epochs = 100
        learning_rate = 2e-4


    if use_MIntRec:
        class_weights = torch.tensor([0.3878, 0.3924, 0.5252, 0.8134, 0.9014, 0.9137, 0.9529, 1.0106, 1.0587,
                1.1702, 1.2827, 1.3078, 1.5512, 1.7553, 1.8528, 1.9057, 2.0844, 2.1516,2.1516, 2.1516], device='cuda:0')
    else:
        class_weights = torch.tensor([1.2717, 0.8185, 0.8395, 1.2489], device='cuda:0')

        # class_weights = torch.tensor([1.2645, 0.8270], device='cuda:0')

    modality = 'av'
    # modality = '40'
    train_dataloader, val_dataloader, test_dataloader = load_data(modality)
    best_i = 0.0
    best_prec = 0.0
    best_f1 = 0.0
    best_wa = 0.0
    best_ua = 0.0
    for i in range(5):
        model = Classifier(modality).to('cuda')
        best_WA, best_precision, best_UA, best_F1 = train(model, train_dataloader, val_dataloader, test_dataloader, class_weights)
        if best_F1 > best_f1:
            best_i = i
            best_prec = best_precision
            best_f1 = best_F1
            best_wa = best_WA
            best_ua = best_UA

    print(f'all_{best_i}: WA/acc:{best_wa} UA/rec:{best_ua} WAF/F1:{best_f1}')
