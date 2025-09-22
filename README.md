## Paper: ADMC: Attention-based Diffusion Model for Missing Modalities Feature Completion

## 🔗 Resources


### 1. Preprocessed IEMOCAP datasets and MIntRec
- *: [Download](https://pan.baidu.com/s/1IGo1cC9IjR2iTyYAOtyS8A?pwd=nw2t), code: nw2t 

## 🛠️ Requirements

```
torch>=2.0
diffusers==0.29.2
```

## 🚀 Training

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


## 📁 Directory Structure

```
├── data/
│ ├── IEMOCAP/
│ └── MIntRec/
├── dataloaders/
│ ├── dataset.py
│ ├── dataset_IEMOCAP.py
│ └── dataset_MIntRec.py
├── models/
│ ├── decoder.py
│ ├── Diffusion.py
│ ├── lstm.py
│ ├── main.py
│ ├── MF.py
│ ├── textcnn.py
│ ├── transformer.py
│ └── Unet_.py
├── outputs/
├── utils/
├── log/
├── opts.py
├── run.py
```

## 📞 Contact

For questions or access to datasets, please contact: \[zhangwei_self@qq.com]

