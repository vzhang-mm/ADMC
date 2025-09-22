from torch.optim import Adam, AdamW
from tqdm import tqdm
from torch import nn
import torch.autograd.profiler as profiler
import torch
from models.main import classifier
from opts import parse_opts
args = parse_opts()
if args.dataset == 'IEMOCAP':
    from datasets.dataset_IEMOCAP import Dataset
elif args.dataset == 'MIntRec':
    from datasets.dataset_MIntRec import Dataset
else:
    from datasets.dataset import Dataset, labels_to_idx
import os
from tensorboardX import SummaryWriter
from datetime import datetime
import socket
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)  # 移动张量到设备
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}  # 字典的每个元素递归调用
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]  # 列表的每个元素递归调用
    elif isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)  # 元组的每个元素递归调用
    return data.to(device)

def compute_metrics(preds, labels):
    preds = torch.stack(preds,dim=0)
    labels = torch.stack(labels,dim=0)
    # 计算总体准确率
    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())#WA
    precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro',zero_division=0)
    recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)#UA
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

    return round(acc,4), round(precision,4), round(recall,4), round(f1,4)

zh = ['va', 'ta', 'tv', 'a', 'v', 't']

def train(model, train_data, val_data, df_test, learning_rate, epochs, batch_size):
    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = DataLoader(train, batch_size, collate_fn=train.collate_fn)##shuffle=True, num_workers=4
    val_dataloader = DataLoader(val, batch_size, collate_fn=val.collate_fn)#默认为False
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    # 定义优化器
    # optimizer = AdamW([
    #     {'params': model.Text_model.parameters(), 'lr': learning_rate*0.1},
    #     {'params': model.Wav_model.parameters(), 'lr': learning_rate},
    #     {'params': model.Vid_model.parameters(), 'lr': learning_rate},
    #     # {'params': model.diffusion.parameters(), 'lr': learning_rate},
    #     {'params': model.mlp_model.parameters(), 'lr': learning_rate},
    #     {'params': model.Fusion_model.parameters(), 'lr': learning_rate},
    #     {'params': model.fc.parameters(), 'lr': learning_rate}
    # ],weight_decay=1e-5)

    # optimizer = AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-5)   # 更优, weight_decay=1e-5 如果你不设置 weight_decay 参数，优化器将只根据损失函数的梯度更新模型参数，而不会施加额外的正则化约束。
    optimizer = Adam(model.parameters(),lr=learning_rate)
    # 动态调整学习率
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # 设置学习率调度器
    num_training_steps = epochs * len(train_dataloader)  # 实际训练步骤数
    num_warmup_steps = int(0.1 * num_training_steps)  # 设置为总训练步骤数的 10%,30epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    if use_cuda:
        model = model.cuda()

    log_dir = os.path.join('./log', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    best_accuracy = 0.5
    if args.dataset == 'IEMOCAP':
        best_mlp_loss = 0.8
    else:
        best_mlp_loss = 1.5

    # 开始进入训练循环
    for epoch_num in range(epochs):
        model.train()
        if args.use_FeatureExtraction:
            if args.use_Fusion:##融合训练时固定特征提取网络
                if args.use_Fusion_GD:
                    model.feature_extraction_model.eval()
        else:
            if args.use_DiffusionMlp:
                model.feature_extraction_model.eval()
            else:
                model.feature_extraction_model.eval()
                model.CDMC_model.eval()

        # 定义两个变量，用于存储训练集的准确率和损失
        total_loss_train = 0.0
        total_preds = []
        total_labels = []
        total_preds_alls = [[] for _ in range(6)]#'va', 'ta', 'tv', 'a', 'v', 't','avt'
        # 进度条函数tqdm
        for samples in tqdm(train_dataloader):
            labels = samples['label'].long().to(device)
            text = to_device(samples['text'], device)
            vid = to_device(samples['vid'], device)
            wav = to_device(samples['audio'], device)
            # 模型更新
            # model.zero_grad()
            optimizer.zero_grad()

            # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
            #     with profiler.record_function("model_inference"):
            # 通过模型得到输出
            if not args.use_DiffusionMlp:#训练特征提取
                preds, c_loss, _ = model(text, vid, wav, labels)
                # print(preds,c_loss)
                batch_loss = c_loss
                total_loss_train += batch_loss.item()
                total_preds.extend(preds)
                total_labels.extend(labels)
            else:#训练扩散模型
                all_preds, all_class_preds, d_loss = model(text, vid, wav, labels)
                batch_loss = d_loss
                total_loss_train += batch_loss.item()

                for i in range(len(all_preds)):  ################每种缺失状态的loss (6*4)
                    total_preds_alls[i].extend(all_preds[i])

            # model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()##
        # scheduler.step()

        model_param_ids = set(id(p) for p in model.feature_extraction_model.parameters())#model.Fusion_model.parameters()
        for param_group in optimizer.param_groups:
            if any(id(param) in model_param_ids for param in param_group['params']):
                print(f'model.feature_extraction_model 的学习率: {param_group["lr"]:.6f}')
                break  # 只需要打印一次就可以了

        if not args.use_DiffusionMlp:
            total_acc_train, train_precision, train_recall, train_f1 = compute_metrics(total_preds, total_labels)
        else:
            total_train_metrics_list = np.mean(np.array(total_preds_alls),axis=1)#每一列求均值

        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_loss_val = 0.0
        val_total_preds = []
        val_total_labels = []
        val_total_preds_alls = [[] for _ in range(6)]# 'va', 'ta', 'tv', 'a', 'v', 't','avt'
        # 不需要计算梯度
        with torch.no_grad():
            model.eval()  # model.eval()的作用是不启用 Batch Normalization 和 Dropout。
            for val_samples in tqdm(val_dataloader):
                val_labels = val_samples['label'].long().to(device)
                val_vid = to_device(val_samples['vid'], device)
                val_text = to_device(val_samples['text'], device)
                val_wav = to_device(val_samples['audio'], device)
                # 通过模型得到输出
                if not args.use_DiffusionMlp:
                    val_preds, val_c_loss, _ = model(val_text, val_vid, val_wav, val_labels)
                    total_loss_val += val_c_loss.item()
                    val_total_preds.extend(val_preds)
                else:
                    val_all_preds, _, val_d_loss = model(val_text, val_vid, val_wav, val_labels)
                    batch_loss = val_d_loss
                    total_loss_val += batch_loss.item()
                    for i in range(len(val_all_preds)):################每种缺失状态的评价指标 (7*4)
                        val_total_preds_alls[i].extend(val_all_preds[i])

                val_total_labels.extend(val_labels)

            if not args.use_DiffusionMlp:
                total_acc_val, val_precision, val_recall, val_f1 = compute_metrics(val_total_preds, val_total_labels)
            else:
                total_val_metrics_list = np.mean(np.array(val_total_preds_alls), axis=1)  # 每一列求均值

        if not args.use_DiffusionMlp:
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_dataloader): .3f} 
              | Train Accuracy: {total_acc_train} 
              | Train Precision: {train_precision} 
              | Train Recall: {train_recall} 
              | Train F1: {train_f1} 

              | Val Loss: {total_loss_val / len(val_dataloader): .3f} 
              | Val Accuracy: {total_acc_val}
              | Val Precision: {val_precision} 
              | Val Recall: {val_recall} 
              | Val F1: {val_f1} ''')

            # 如果当前测试集准确率高于之前的最佳结果，则保存模型
            if total_acc_val > best_accuracy:
            # if 0.79 > total_acc_val > 0.78:
                best_accuracy = total_acc_val
                if args.dataset == 'MIntRec':
                    if args.use_text:
                        torch.save(model.feature_extraction_model.netL.state_dict(), "best_MI_text.pth")
                    elif args.use_wav:
                        torch.save(model.feature_extraction_model.netA.state_dict(), "best_MI_wav.pth")
                    elif args.use_vid:
                        torch.save(model.feature_extraction_model.netV.state_dict(), "best_MI_vid.pth")
                    else:
                        torch.save(model.feature_extraction_model.state_dict(), "best_MI_fusion.pth")
                elif args.dataset == 'IEMOCAP':
                    if args.use_text:
                        torch.save(model.feature_extraction_model.netL.state_dict(), "best_IE_text.pth")
                    elif args.use_wav:
                        torch.save(model.feature_extraction_model.netA.state_dict(), "best_IE_wav.pth")
                    elif args.use_vid:
                        torch.save(model.feature_extraction_model.netV.state_dict(), "best_IE_vid.pth")
                    else:
                        torch.save(model.feature_extraction_model.state_dict(), "best_IE_fusion.pth")
                else:
                    if args.use_text:
                        torch.save(model.feature_extraction_model.Text_model.state_dict(), "best_MI_text.pth")
                    elif args.use_wav:
                        torch.save(model.feature_extraction_model.Wav_model.state_dict(), "best_MI_wav.pth")
                    elif args.use_vid:
                        torch.save(model.feature_extraction_model.Vid_model.state_dict(), "best_MI_vid.pth")
                    else:
                        torch.save(model.feature_extraction_model.state_dict(), "best_MI_fusion.pth")

                print(f"Epoch {epoch_num+1}: Val accuracy improved to {best_accuracy}, model saved.")
                #评估
                # evaluate(model, df_test, batch_size, args.use_DiffusionMlp)

            writer.add_scalar('Train Accuracy', total_acc_train, epoch_num)
            writer.add_scalar('Val Accuracy', total_acc_val, epoch_num)

        else:#不计算acc
            print('Epoch {}'.format(epoch_num + 1))
            print('Train Loss: {}'.format(round(total_loss_train / len(train_dataloader), 3)))
            for i in range(len(total_train_metrics_list)):
                print('Train:{},loss,{}'.format(zh[i],total_train_metrics_list[i]))
            print('Val Loss: {}'.format(round(total_loss_val / len(val_dataloader), 3)))
            for i in range(len(total_val_metrics_list)):#6
                print('Val:{},loss,{}'.format(zh[i],total_val_metrics_list[i]))

            #保存
            if round(total_loss_val / len(val_dataloader), 3) < best_mlp_loss:
                best_mlp_loss = round(total_loss_val / len(val_dataloader), 3) - 0.01
                if args.dataset == 'MIntRec':
                    #只保存diff部分网络参数
                    if args.use_Fusion:
                        torch.save(model.CDMC_model.state_dict(), "best_MI_Fusion_mlp.pth")
                    else:
                        torch.save(model.CDMC_model.state_dict(), "best_MI_mlp.pth")
                else:
                    if args.use_Fusion:
                        torch.save(model.CDMC_model.state_dict(), "best_IE_Fusion_mlp.pth")
                    else:
                        torch.save(model.CDMC_model.state_dict(), "best_IE_mlp.pth")
                print(f"Epoch {epoch_num+1}: Val loss improved to {round(total_loss_val / len(val_dataloader), 3)}, model saved.")

                # 评估，最后效果才评估
                # evaluate(model, df_test, batch_size, args.use_DiffusionMlp)
        #要求每epoch都评估
        evaluate(model, df_test, batch_size, args.use_DiffusionMlp)
        
        writer.add_scalar('train_loss_epoch', round(total_loss_train / len(train_dataloader), 3), epoch_num)
        writer.add_scalar('val_loss_epoch', round(total_loss_val / len(val_dataloader), 3), epoch_num)

        if epoch_num< 3:
            # model.feature_extraction_model.save_data_epoch(epoch_num)
            if args.use_DiffusionMlp:
                # 保存每个epoch的特征值
                model.feature_extraction_model.save_data_epoch(epoch_num)
            else:
                if not args.use_FeatureExtraction:
                    # 保存每个epoch的生成特征值
                    model.save_fake_features_epoch(epoch_num)

        # 打印 profiler 结果
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # prof.export_chrome_trace("trace.json")


# 测试模型
def evaluate(model, test_data, batch_size, use_DiffusionMlp):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, collate_fn=test.collate_fn)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    test_total_preds = []
    test_total_labels = []
    test_total_preds_alls = [[] for _ in range(6)]
    total_loss_test = 0
    with torch.no_grad():
        model.eval()
        for test_samples in test_dataloader:
            test_labels = test_samples['label'].long().to(device)
            test_text = to_device(test_samples['text'], device)
            test_vid = to_device(test_samples['vid'], device)
            val_wav = to_device(test_samples['audio'], device)

            if not use_DiffusionMlp:
                test_preds, test_c_loss, _ = model(test_text, test_vid, val_wav, test_labels)
                test_total_preds.extend(test_preds)
            else:
                test_all_preds, _, test_d_loss = model(test_text, test_vid, val_wav, test_labels)
                batch_loss = test_d_loss
                total_loss_test += batch_loss.item()
                for i in range(len(test_all_preds)):  ################每种缺失状态的评价指标 (6*4)
                    test_total_preds_alls[i].extend(test_all_preds[i])
            test_total_labels.extend(test_labels)

        if not use_DiffusionMlp:
            total_acc_test, test_precision, test_recall, test_f1 = compute_metrics(test_total_preds, test_total_labels)
            print(
                f'''
              | Test Accuracy: {total_acc_test} 
              | Test Precision: {test_precision} 
              | Test Recall: {test_recall} 
              | Test F1: {test_f1} ''')
        else:
            print('Test Loss: {}'.format(round(total_loss_test / len(test_dataloader), 3)))
            total_test_metrics_list = np.mean(np.array(test_total_preds_alls), axis=1)  # 每一列求均值

            for i in range(len(total_test_metrics_list)):#6
                print('Test:{},loss,{}'.format(zh[i],total_test_metrics_list[i]))


if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    import pickle
    import sys
    from sklearn.utils.class_weight import compute_class_weight
    import random
    from utils import functions

    # def set_seed(seed):
    #     random.seed(seed)  # Python random 模块
    #     np.random.seed(seed)  # NumPy
    #     torch.manual_seed(seed)  # PyTorch CPU
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)  # PyTorch GPU
    #         torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    #     # 为了确保使用cudnn后端的可重复性
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # # 设置随机数种子
    # set_seed(42)

    if args.dataset == 'IEMOCAP':
        with open('./data/IEMOCAP/1/train_IEMOCAP.pkl', 'rb') as file:  # /root/autodl-tmp/MIntRec
            train_data = pickle.load(file)
        with open('./data/IEMOCAP/1/val_IEMOCAP.pkl', 'rb') as file:#a,v,t,av,at,vt
            val_data = pickle.load(file)#test_data
        with open('./data/IEMOCAP/1/test_IEMOCAP.pkl', 'rb') as file:
            test_data = pickle.load(file)#
        labels = [i_dict['label'] for i_dict in train_data]
    elif args.dataset == 'MIntRec':
        with open('./data/MIntRec/train_MIntRec.pkl', 'rb') as file:  # /root/autodl-tmp/MIntRec
            train_data = pickle.load(file)
        with open('./data/MIntRec/val_MIntRec.pkl', 'rb') as file:  # a,v,t,av,at,vt
            val_data = pickle.load(file)  # test_data
        with open('./data/MIntRec/test_MIntRec.pkl', 'rb') as file:
            test_data = pickle.load(file)  #
        labels = [i_dict['label'] for i_dict in train_data]
        # labels = [labels_to_idx[i_dict['label']] for i_dict in train_data]

    # 计算每个类别的权重
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda')
    print('class_weights',class_weights)

    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    df_test = pd.DataFrame(test_data)

    print(len(df_train), len(df_val), len(df_test))

    EPOCHS = args.nEpochs
    batch_size = args.batch_size
    LR = args.lr

    model = classifier(class_weights = class_weights)

    s_dict = model.state_dict()
    layer_name = []

    if args.use_FeatureExtraction:
        print('use_FeatureExtraction')
        if args.use_Fusion:
            if args.use_Fusion_GD:
                s_dict, layer_name = functions.load_checkpoint_FE(s_dict, layer_name)
                # # 加载并固定
                model = functions.load_GD(model, s_dict, layer_name)
                print('固定特征提取网络参数，融合训练分类层')
            else:
                s_dict, layer_name = functions.load_checkpoint_fusion_FE(s_dict, layer_name)
                model = functions.load_GD(model, s_dict, layer_name)
                print('不固定特征提取网络参数，融合训练')

        else:#单模态训练
            # 独立加载个模态特征提取参数
            # s_dict, layer_name = functions.load_checkpoint_FE(s_dict, layer_name)
            # # 加载并固定
            # model = functions.load_GD(model, s_dict, layer_name)

            print('单模态训练,从头开始训练')

        train(model, df_train, df_val, df_test, LR, EPOCHS, batch_size)
        torch.save(model.state_dict(), "bertMy.pth")
 
    else:###########################################模态缺失、扩散模型、随机噪声填充验证
        if args.use_DiffusionMlp:
            print('训练扩散模型MLP网络')

            if args.use_Fusion:
                print('训练融合特征提取的扩散模型MLP网络')
                # 加载融合训练特征提取参数
                s_dict, layer_name = functions.load_checkpoint_fusion_FE(s_dict, layer_name)
            else:
                print('训练独立特征提取的扩散模型MLP网络')
                #独立加载个模态特征提取参数
                s_dict, layer_name = functions.load_checkpoint_FE(s_dict, layer_name)

            # s_dict, layer_name = functions.load_checkpoint_MLP(s_dict, layer_name)#加载MLP

            # 加载并固定
            model = functions.load_GD(model, s_dict, layer_name)

        else:#训练模态缺失
            if args.use_zero:
                print("使用0补全")
                print('use_modality',args.use_modality)
                # 加载特征提取器
                s_dict, layer_name = functions.load_checkpoint_FE(s_dict, layer_name)

                # 加载并固定
                model = functions.load_GD(model, s_dict, layer_name)

            elif args.use_MMIR or args.use_MEIR:#MMIR,模态缺失
                print("使用MLP重建缺失模态特征")
                print('use_modality',args.use_modality)

                if args.use_Fusion:
                    print('使用融合训练的特征提取网络')
                    # 加载融合训练特征提取参数
                    s_dict, layer_name = functions.load_checkpoint_fusion_FE(s_dict, layer_name)
                else:
                    print('使用独立训练的特征提取网络')
                    #独立加载个模态特征提取参数
                    s_dict, layer_name = functions.load_checkpoint_FE(s_dict, layer_name)
                # 加载MLP
                s_dict, layer_name = functions.load_checkpoint_MLP(s_dict, layer_name)

                # 加载并固定
                # for name in s_dict:
                #     if 'mlp_model' in name:
                #         layer_name.append(name)
                model = functions.load_GD(model, s_dict, layer_name)

            else:
                print("没有选择训练模式")
                sys.exit(1)

        train(model, df_train, df_val, df_test, LR, EPOCHS, batch_size)
        torch.save(model.state_dict(), "bestMy.pth")





#单独训练每个模态特征提取网络
# python run.py --nEpochs 120 --batch_size 160 --lr 1e-4 --use_FeatureExtraction --dataset=MIntRec --use_text
#融合训练
# python run.py --nEpochs 120 --batch_size 320 --lr 1e-4 --use_FeatureExtraction --dataset=MIntRec --use_Fusion

#训练Diffusion
# python run.py --nEpochs 200 --batch_size 320 --lr 1e-3 --use_DiffusionMlp  --dataset=IEMOCAP --use_full
# python run.py --nEpochs 200 --batch_size 160 --lr 1e-3 --use_DiffusionMlp  --dataset=MIntRec --use_full

#use_zero
# python run.py --nEpochs 100 --batch_size 320 --lr 5e-5 --dataset=IEMOCAP --use_modality=4  --use_zero

#MMIR: t->tva
# python run.py --nEpochs 120 --batch_size 320 --lr 5e-5 --use_MMIR --dataset=IEMOCAP --use_modality=0 --N=50  --use_full
# python run.py --nEpochs 120 --batch_size 160 --lr 5e-5 --use_MMIR --dataset=MIntRec --use_modality=5 --N=50  --use_full

# MEIR
# python run.py --nEpochs 120 --batch_size 320 --lr 5e-5 --use_MEIR --dataset=MIntRec --use_full --N=50
# Fusion下MEIR
# python run.py --nEpochs 120 --batch_size 320 --lr 5e-5 --use_MEIR --dataset=MIntRec --use_full --use_Fusion --N=50

