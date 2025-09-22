
# 全部流程代码
import numpy as np
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
import h5py

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.labels = [label for label in df['label']]
        self.texts = [text for text in df['text']]
        self.vids = [vid for vid in df['vid']]
        self.wavs = [wav for wav in df['wav']]

        self.mean_std = h5py.File('F:/数据集/IEMOCAP/IEMOCAP_features_2021/A/comparE_mean_std.h5', 'r')
        # 使用第一个数据集的均值
        self.mean = torch.from_numpy(self.mean_std['1']['mean'][()]).unsqueeze(0).float()
        self.std = torch.from_numpy(self.mean_std['1']['std'][()]).unsqueeze(0).float()

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return torch.tensor(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx].astype(np.float32)

    def get_batch_vids(self, idx):
        return self.vids[idx].astype(np.float32)

    def get_batch_wavs(self, idx):
        return self.wavs[idx].astype(np.float32)


    def __getitem__(self, idx):
        samples = self.load_samples(idx)
        return samples

    def load_samples(self,idx):
        batch_y = self.get_batch_labels(idx)

        batch_texts = self.get_batch_texts(idx)
        batch_vids = self.get_batch_vids(idx)
        batch_wavs = self.get_batch_wavs(idx)

        return {
            'text': {'texts': batch_texts},
            'audio': {'wavs': batch_wavs},
            'vid': {'vids': batch_vids},
            'label': batch_y,
        }

    def pad_and_mask(self, data_list, max_len=512, padding_value=0.0):
        sequences = [torch.tensor(data) if data is not None else torch.empty(0, data_list[0].shape[1]) for data in data_list]
        sequences = [seq[:max_len] for seq in sequences]  # Truncate to max_len
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        lengths = [min(seq.shape[0], max_len) for seq in sequences]
        masks = torch.zeros(padded_sequences.shape[:2], dtype=torch.float32)
        for i, length in enumerate(lengths):
            masks[i, :length] = 1

        return padded_sequences, masks

    def collate_fn(self, batch):
        # 处理文本部分
        texts, text_masks = self.pad_and_mask([item['text']['texts'] for item in batch], padding_value=0.0)

        vids, vid_masks = self.pad_and_mask([item['vid']['vids'] for item in batch], padding_value=0.0)

        wavs, wav_masks = self.pad_and_mask([item['audio']['wavs'] for item in batch], padding_value=0.0)


        labels = torch.tensor([item['label'] for item in batch])

        wavs = self.normalize_on_utt(wavs)
        # wavs = self.normalize_on_trn(wavs)

        return {
            'text': {'texts': texts, 'text_masks':text_masks},
            'vid': {'vids': vids, 'vid_masks': vid_masks},
            'audio': {'wavs': wavs, 'wav_masks': wav_masks},
            'label': labels
        }


    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features

    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

if __name__ == '__main__':
    import pandas as pd
    with open('D:/Desktop/MMIR/data/IEMOCAP/test_IEMOCAP.pkl', 'rb') as file:
        val_data = pickle.load(file)

    df_val = pd.DataFrame(val_data)

    val = Dataset(df_val)
    val_dataloader = torch.utils.data.DataLoader(val, 10, collate_fn=val.collate_fn)

    for val_samples in val_dataloader:
        print(type(val_samples['text']))
        print(val_samples['text'])
        print(val_samples['vid']['vids'].shape)

        print(val_samples['audio']['wavs'].shape)

        print(val_samples['label'].shape)

        break