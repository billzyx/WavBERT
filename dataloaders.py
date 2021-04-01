from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import core
import os
import re


class ADReSSTextDataset(Dataset):
    def get_file_text(self, file_path, level_list, punctuation_list):
        return core.load_file(file_path, level_list, punctuation_list)

    def get_audio_embedding(self, file_path):
        file_path = file_path.replace('asr_text', 'asr_embedding').replace('.txt', '.npy')
        return np.load(file_path)


class ADReSSTextTrainDataset(ADReSSTextDataset):
    def __init__(self, dir_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, sentiment in (('cc', 0), ('cd', 1)):
            folder = os.path.join(dir_path, 'asr_text', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = self.get_file_text(file_path, level_list, punctuation_list)
                text_file['audio_embedding'] = self.get_audio_embedding(file_path)
                if len(text_file['text'].split()) > filter_min_word_length:
                    self.X.append(text_file)
                    self.Y.append(sentiment)
                    self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                    self.file_idx.append(name.split('.')[0])

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        with open(os.path.join(dir_path, 'cc_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = lines[i].split(';')[3].strip()
                if file_label == 'NA':
                    file_label = 29
                else:
                    file_label = int(file_label)
                labels[file_id] = file_label
        with open(os.path.join(dir_path, 'cd_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                labels[file_id] = file_label
        print(labels)
        return labels


class ADReSSo21TextTrainDataset(ADReSSTextDataset):
    def __init__(self, dir_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, sentiment in (('cn', 0), ('ad', 1)):
            folder = os.path.join(dir_path, 'asr_text', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = self.get_file_text(file_path, level_list, punctuation_list)
                text_file['audio_embedding'] = self.get_audio_embedding(file_path)
                if len(text_file['text'].split()) > filter_min_word_length:
                    self.X.append(text_file)
                    self.Y.append(sentiment)
                    self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                    self.file_idx.append(name.split('.')[0])

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        data = pd.read_csv(os.path.join(dir_path, 'adresso-train-mmse-scores.csv'))
        df = pd.DataFrame(data)

        for index, row in df.iterrows():
            # print(row[0], row['mmse'])
            labels[row['adressfname']] = int(row['mmse'])
        print(labels)
        return labels


class ADReSSTextTestDataset(ADReSSTextDataset):
    def __init__(self, dir_path, label_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        labels, mmse_labels = self.load_test_label(label_path)
        for name in tqdm(sorted(os.listdir(os.path.join(dir_path, 'asr_text')))):
            file_path = os.path.join(dir_path, 'asr_text', name)
            text_file = self.get_file_text(file_path, level_list, punctuation_list)
            text_file['audio_embedding'] = self.get_audio_embedding(file_path)
            if len(text_file['text'].split()) > filter_min_word_length:
                self.X.append(text_file)
                self.Y.append(labels[name.split('.')[0]])
                self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                self.file_idx.append(name.split('.')[0])

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_test_label(self, label_path):
        labels = {}
        mmse_labels = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                file_mmse_label = int(lines[i].split(';')[4].strip())
                labels[file_id] = file_label
                mmse_labels[file_id] = file_mmse_label
        print(labels)
        print(mmse_labels)
        return labels, mmse_labels


class ADReSSTextTranscriptDataset(Dataset):
    def get_file_text(self, file_path):
        text_file = ''
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                text = line.strip().replace('	', ' ')
                if line.startswith('*'):
                    text = text.split(':', maxsplit=1)[1] + ' '
                    temp_idx = idx
                    while not '' in lines[temp_idx]:
                        temp_idx += 1
                        text += lines[temp_idx].strip() + ' '
                    text = text.split('')[0]
                    text_file += text

        # print(text_file)
        text_file = text_file.replace('_', ' ')
        text_file = re.sub(r'\[[^\]]+\]', '', text_file)
        text_file = re.sub('[^0-9a-zA-Z,. \'?]+', '', text_file)
        text_file = text_file.replace('...', '').replace('..', '')
        # print(text_file)
        return text_file


class ADReSSTextTranscriptTrainDataset(ADReSSTextTranscriptDataset):
    def __init__(self, dir_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, ad_label in (('cc', 0), ('cd', 1)):
            folder = os.path.join(dir_path, 'transcription', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = self.get_file_text(file_path)
                self.X.append(text_file)
                self.Y.append(ad_label)
                self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                self.file_idx.append(name.split('.')[0])

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        with open(os.path.join(dir_path, 'cc_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = lines[i].split(';')[3].strip()
                if file_label == 'NA':
                    file_label = 29
                else:
                    file_label = int(file_label)
                labels[file_id] = file_label
        with open(os.path.join(dir_path, 'cd_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                labels[file_id] = file_label
        print(labels)
        return labels


class ADReSSTextTranscriptTestDataset(ADReSSTextTranscriptDataset):
    def __init__(self, dir_path, label_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        labels, mmse_labels = self.load_test_label(label_path)
        for name in tqdm(sorted(os.listdir(os.path.join(dir_path, 'transcription')))):
            file_path = os.path.join(dir_path, 'transcription', name)
            text_file = self.get_file_text(file_path)
            self.X.append(text_file)
            self.Y.append(labels[name.split('.')[0]])
            self.Y_mmse.append(mmse_labels[name.split('.')[0]])
            self.file_idx.append(name.split('.')[0])

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_test_label(self, label_path):
        labels = {}
        mmse_labels = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                file_mmse_label = int(lines[i].split(';')[4].strip())
                labels[file_id] = file_label
                mmse_labels[file_id] = file_mmse_label
        print(labels)
        print(mmse_labels)
        return labels, mmse_labels


class ADReSSo21TextProgressionTrainDataset(ADReSSTextDataset):
    def __init__(self, dir_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.file_idx = []
        for folder, sentiment in (('no_decline', 0), ('decline', 1)):
            folder = os.path.join(dir_path, 'asr_text', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = self.get_file_text(file_path, level_list, punctuation_list)
                text_file['audio_embedding'] = self.get_audio_embedding(file_path)
                if len(text_file['text'].split()) > filter_min_word_length:
                    self.X.append(text_file)
                    self.Y.append(sentiment)
                    self.file_idx.append(name.split('.')[0])

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': 0.0,
        }

    def __len__(self):
        return len(self.X)

