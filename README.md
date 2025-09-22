# 📄 Paper: **ADMC — Attention-based Diffusion Model for Missing Modalities Feature Completion**

## 🔗 Resources

### 1. Preprocessed IEMOCAP & MIntRec Datasets

* [Download Link](https://pan.baidu.com/s/1IGo1cC9IjR2iTyYAOtyS8A?pwd=nw2t) (code: `nw2t`)
* 包含已处理的 **AVEC2013/2014**, **IEMOCAP**, 和 **MIntRec** 数据集，便于快速复现实验。

---

## 🛠️ Requirements

```bash
torch>=2.0
diffusers==0.29.2
```

---

## 🚀 Training

我们提供多种实验配置以覆盖 **特征提取、扩散补全** 以及 **MMIR / MEIR**方法。以下为常用运行示例：

### 1. 单模态特征提取

```bash
python run.py --nEpochs 120 --batch_size 160 --lr 1e-4 \
              --use_FeatureExtraction --dataset=MIntRec --use_text
```

### 2. 多模态融合训练

```bash
python run.py --nEpochs 120 --batch_size 320 --lr 1e-4 \
              --use_FeatureExtraction --dataset=MIntRec --use_Fusion
```

### 3. 扩散模型 (Diffusion)

```bash
# IEMOCAP
python run.py --nEpochs 200 --batch_size 320 --lr 1e-3 \
              --use_DiffusionMlp --dataset=IEMOCAP --use_full

# MIntRec
python run.py --nEpochs 200 --batch_size 160 --lr 1e-3 \
              --use_DiffusionMlp --dataset=MIntRec --use_full
```

### 4. Zero-shot 设定

```bash
python run.py --nEpochs 100 --batch_size 320 --lr 5e-5 \
              --dataset=IEMOCAP --use_modality=4 --use_zero
```

### 5. MMIR: *t → tva*

```bash
# IEMOCAP
python run.py --nEpochs 120 --batch_size 320 --lr 5e-5 \
              --use_MMIR --dataset=IEMOCAP --use_modality=0 --N=50 --use_full

# MIntRec
python run.py --nEpochs 120 --batch_size 160 --lr 5e-5 \
              --use_MMIR --dataset=MIntRec --use_modality=5 --N=50 --use_full
```

### 6. MEIR

```bash
# 单独 MEIR
python run.py --nEpochs 120 --batch_size 320 --lr 5e-5 \
              --use_MEIR --dataset=MIntRec --use_full --N=50

# 融合下 MEIR
python run.py --nEpochs 120 --batch_size 320 --lr 5e-5 \
              --use_MEIR --dataset=MIntRec --use_full --use_Fusion --N=50
```

---

## 📁 Directory Structure

```
├── data/
│   ├── IEMOCAP/
│   └── MIntRec/
├── dataloaders/
│   ├── dataset.py
│   ├── dataset_IEMOCAP.py
│   └── dataset_MIntRec.py
├── models/
│   ├── decoder.py         # 注意力模块
│   ├── Diffusion.py       # 扩散模型核心
│   ├── lstm.py            # LSTM 模块
│   ├── main.py            # 主入口/调度
│   ├── MF.py              # 多模态融合 (MF) 模型
│   ├── textcnn.py         # 文本 CNN 模块
│   ├── transformer.py     # Transformer 模块
│   └── Unet_.py           # UNet 模块
├── outputs/               # 实验结果与生成文件
├── utils/                 # 工具函数
├── log/                   # 训练日志
├── opts.py                # 参数配置
├── run.py                 # 主运行脚本
```

---

## 📞 Contact

如有问题或需获取数据，请联系：
📧 [zhangwei\_self@qq.com](mailto:zhangwei_self@qq.com)
