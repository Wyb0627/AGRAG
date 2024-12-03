import os

import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer, BertModel
import json
import pandas as pd
import random
import re

# t0_label = ['CS', 'Medical', 'Civil']
# self.label2id={label:idx for idx, label in enumerate(t0_label)}
label_transform = {'cs': 'computer science',
                   'ece': 'electrical and computer engineering',
                   'mae': 'mechanical and aerospace engineering',
                   'civil': 'civil engineering'}


class WOS4RoundDataset(Dataset):
    def __init__(self, file_path: str, label2id: dict):
        self.file_path = file_path
        self.label2id = label2id
        self.total_data, self.total_label, self.total_label_id = self.load_data()

    def load_data(self):
        total_data = []
        total_label = []
        total_label_id = []
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.split('\t')[0].lower()
                text_list = line.split('\t')[1:]
                if label == 'electrical generator':
                    continue
                label = label.replace("'s", '')
                label_list = re.findall(r'(?u)\b\w\w+\b', label)
                cleaned_label = str(' '.join(label_list))
                cleaned_label = cleaned_label.replace('_', ' ')
                total_label.append(cleaned_label)
                if cleaned_label == 'hepatitis c':
                    cleaned_label = 'hepatitis'
                total_label_id.append(self.label2id[cleaned_label])
                total_data.append(' '.join(text_list))
        return total_data, total_label, total_label_id

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        item = {"doc_token": self.total_data[idx],
                'label_#': self.total_label_id[idx],
                'label': self.total_label[idx]}
        return item


class ReutersDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.total_data, self.total_label = self.load_data()

    def load_data(self):
        total_data = []
        total_label = []
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.split('\t')[0].lower()
                text_list = line.split('\t')[1:]
                total_label.append(label)
                total_data.append(' '.join(text_list))
        return total_data, total_label

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        item = {"doc_token": self.total_data[idx],
                'label': self.total_label[idx]}
        return item


class FewShotTemporalDataset(Dataset):
    def __init__(self, dataset: str, split: str, label2id: dict, shot_num: int, shot_label: list[str], seed: int = 0):
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.split = split
        self.label2id = label2id
        self.shot_num = shot_num
        self.shot_label = shot_label  # ['CS-Bioinformatics', ...] or ['CS', 'Medical', 'Civil', ...]
        self.total_data, self.total_label_t0, self.total_label_t1 = self.load_data()

    def load_data(self) -> list[dict]:
        total_data = []
        total_label_t0 = []
        total_label_t1 = []
        for label_name in self.shot_label:
            for filename in os.listdir(f'../dataset/{self.dataset}/{self.dataset}_by_label/{label_name}'):
                sub_label_name = filename.split('.json')[0]
                with open(f'../dataset/{self.dataset}/{self.dataset}_by_label/{label_name}/{filename}') as f:
                    data_list = json.load(f)
                    if self.split == 'train':
                        random_nums = np.random.randint(0, 50, self.shot_num)
                        for idx in random_nums:
                            total_data.append(data_list[idx])
                            if label_name.lower() in label_transform:
                                label_name_converted = label_transform[label_transform[label_name.lower()]]
                                total_label_t0.append(label_name_converted.lower())
                            else:
                                total_label_t0.append(label_name.lower())
                            if sub_label_name.lower() in label_transform:
                                sub_label_name_converted = label_transform[label_transform[sub_label_name.lower()]]
                                total_label_t1.append(sub_label_name_converted.lower())
                            else:
                                total_label_t1.append(sub_label_name.lower())
        return total_data, total_label_t0, total_label_t1

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        item = {"doc_token": self.total_data[idx],
                'label_t0_#': self.label2id[self.total_label_t0[idx]],
                'label_t1_#': self.label2id[self.total_label_t1[idx]],
                'label_t0': self.total_label_t0[idx],
                'label_t1': self.total_label_t1[idx]}
        return item


class TemporalDatasetLLM(Dataset):
    def __init__(self, dataset: str, split: str, label2id: dict):
        self.dataset = dataset
        self.split = split
        self.label2id = label2id
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.total_data = self.load_data()

    def load_data(self) -> list[dict]:
        total_data = []
        count = 0
        id2data = {}
        with open(f'../dataset/{self.dataset}/{self.dataset}_{self.split}.json') as f:
            for line in f.readlines():
                data = json.loads(line)
                if data['doc_label'][1].lower() == 'electrical generator':
                    continue
                id2data[count] = data
                if data['doc_label'][0].lower() in label_transform:
                    data['doc_label'][0] = label_transform[data['doc_label'][0].lower()]
                label_t0_list = re.findall(r'(?u)\b\w\w+\b', data['doc_label'][0])
                label_t0 = str(' '.join(label_t0_list))
                label_t1_list = re.findall(r'(?u)\b\w\w+\b', data['doc_label'][1])
                label_t1 = str(' '.join(label_t1_list))
                label_t1.replace('-', ' ')
                label_t0.replace('-', ' ')
                data['label_t0'] = label_t0.lower()
                data['label_t1'] = label_t1.lower()
                data['label_t0_#'] = self.label2id[data['label_t0']]
                data['label_t1_#'] = self.label2id[data['label_t1']]
                data['doc_id'] = count
                encodings = self.tokenizer(data['doc_token'],
                                           padding='max_length',
                                           truncation=True,
                                           max_length=512,
                                           return_tensors='pt')
                for key, value in encodings.items():
                    data[key] = value
                total_data.append(data)
                count += 1
        # print(list(total_data[0].keys()))
        # if self.split == 'test':
        #     with open(f'./id2data_{self.dataset}_{self.split}.json', 'w') as f:
        #        f.write(json.dumps(id2data, indent=4))
        return total_data

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        item = {key: value for key, value in self.total_data[idx].items() if
                key not in ['doc_topic', 'doc_keyword', 'doc_label']}
        return item


class WeBankLLM(Dataset):
    def __init__(self, file_dir: str = '../dataset/webank/webank_utf8.csv'):
        self.file_dir = file_dir
        self.total_data = self.load_data()

    def load_data(self):
        total_data = {}
        self.id2label, self.label2id = {}, {}
        data = pd.read_csv(self.file_dir, index_col=False)
        total_data['idx'] = list(data.IDX)
        total_data['code'] = list(data.Code)
        total_data['desc'] = list(data.Desc)
        total_data['label'] = list(data.Label)
        for id, label in enumerate(data.Label.unique()):
            self.id2label[id] = label
            self.label2id[label] = id
        with open(f'../dataset/webank/id2data_webank.json', 'w') as f:
            f.write(json.dumps(self.id2label, indent=4, ensure_ascii=False))
        return total_data

    def __len__(self):
        return len(self.total_data['idx'])

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.total_data.items()}
        return item


def create_icl_samples_label(dataset: str, label2id: dict):
    sample_dict = {}
    with open(f'../dataset/{dataset}/{dataset}_train.json') as f:
        for line in f.readlines():
            data = json.loads(line)
            label_t0_num = label2id[data['doc_label'][0].lower()]
            label_t1_num = label2id[data['doc_label'][1].lower()]
            if label_t0_num not in sample_dict:
                sample_dict[label_t0_num] = data['doc_token']
            else:
                if label_t1_num not in sample_dict:
                    sample_dict[label_t1_num] = data['doc_token']
    sample_list = [sample_dict[i] for i in range(len(sample_dict))]
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        doc_encoding = bert_model(**bert_tokenizer(sample_list,
                                                   padding='max_length',
                                                   truncation=True,
                                                   max_length=512,
                                                   return_tensors='pt'),
                                  output_hidden_states=False).last_hidden_state[:, 0, :]
    return sample_dict, doc_encoding


def icl_doc_encoding(dataset: Dataset, batch_size: int = 128):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    doc_encodings = []
    doc_tokens = []
    doc_labels = []
    with torch.no_grad():
        for item in tqdm.tqdm(dataloader, desc='Doc Encoding'):
            doc_encoding = bert_model(**bert_tokenizer(item['doc_token'],
                                                       padding='max_length',
                                                       truncation=True,
                                                       max_length=512,
                                                       return_tensors='pt'),
                                      output_hidden_states=False).last_hidden_state[:, 0, :]
            # print(doc_encoding.shape)
            doc_encodings.append(doc_encoding)
            doc_tokens.extend(item['doc_token'])
            doc_labels.extend(zip(item['label_t0'], item['label_t1']))
        doc_encodings = torch.cat(doc_encodings, dim=0)
    # print(doc_encodings.shape)
    return doc_encodings, doc_tokens, doc_labels
