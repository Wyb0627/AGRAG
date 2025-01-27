import torch
import os
import random
import numpy as np
import requests
import tqdm
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


def tfidf_predict_old(vectorizer, matrix, query, doc_idx):
    # feature_names = vectorizer.get_feature_names_out()
    keyword_splitted = query.split(' ')
    tfidf_list = []
    tfidf_score = 1
    for keyword_tok in keyword_splitted:
        idx = vectorizer.vocabulary_[keyword_tok]
        tfidf_list.append(matrix.toarray()[doc_idx][idx])
    for score in tfidf_list:
        tfidf_score *= score
    return tfidf_score


def check_device():
    torch.cuda.is_available()  # 查看cuda是否可用
    # torch.cuda.device_count()  # 返回GPU数目
    torch.cuda.get_device_name(0)  # 返回GPU名称，设备索引默认从0开始
    # torch.cuda.current_device()
    print('current device:')
    print(torch.cuda.current_device())
    print('device count:')
    print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def wikidata_search(text: str):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': text,  # 搜索文本
        'language': 'en',  # 查询语言（英文）
        'type': 'item',
        'limit': 1  # 返回最大数目
    }
    # 访问
    get = requests.get(url=url, params=params)
    # 转为json数据
    re_json = get.json()
    # print(re_json)
    print(re_json)
    if re_json['search'] and 'description' in re_json['search'][0]:
        description = re_json['search'][0]['description']
    else:
        description = ''
    return description


def replace_case(old, new, text):
    index = text.lower().find(old.lower())
    if index == -1:
        return text
    return replace_case(old, new, text[:index] + new + text[index + len(old):])


def entity_linking_bert(keyword_list: list[str], target_entity: list[str], target_entity_emb, model, tokenizer):
    id2entity = {idx: entity for idx, entity in enumerate(target_entity)}
    with torch.no_grad():
        keyword_encoding = model(**tokenizer(keyword_list,
                                             padding='longest',
                                             truncation=True,
                                             return_tensors='pt'),
                                 output_hidden_states=False).last_hidden_state[:, 0, :]
        # entity_encoding = model(**tokenizer(target_entity,
        #                                     padding='longest',
        #                                    truncation=True,
        #                                     return_tensors='pt'),
        #                        output_hidden_states=False).last_hidden_state[:, 0, :]
        # Calculate the dot product of the keyword and entity encoding
        dot_product = torch.mm(keyword_encoding, target_entity_emb.T)
        similar_entity_id = torch.argmax(dot_product, dim=1).cpu().tolist()
    similar_entity = [id2entity[idx] for idx in similar_entity_id]
    return similar_entity


def process_keyword(keyword: str):
    keyword = keyword.lower()
    if 'extract' in keyword and 'keyword' in keyword:
        return ''
    keyword = keyword.strip()
    keyword = keyword.strip('.')
    keyword = keyword.strip(',')
    keyword = re.sub(r"\n", "", keyword)
    keyword = re.sub(r":", "", keyword)
    keyword = keyword.strip()
    return keyword


def dataset_to_txt(dataloader, out_put_dir='./input_data'):
    for item in tqdm.tqdm(dataloader, desc=f'Output to dir {out_put_dir}'):
        doc_text = item['doc_token'][0]
        doc_id = item['doc_id'][0]
        domain = item['label_t0'][0]
        area = item['label_t1'][0]
        text = f"This paper is from domain: {domain}, area: {area}. {doc_text}"
        area = area.replace('/', '_')
        with open(f'{out_put_dir}/{doc_id}_{domain}_{area}.txt', 'w') as file:
            file.write(text)


# text = wikidata_search('electrical and computer engineering')
# print(text)


def cos_sim(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


def compute_prefix(pattern):
    prefix = [0] * len(pattern)
    length = 0
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            prefix[i] = length
            i += 1
        else:
            if length != 0:
                length = prefix[length - 1]
            else:
                prefix[i] = 0
                i += 1

    return prefix


def kmp_search(text, pattern):
    prefix = compute_prefix(pattern)
    matches = []
    i, j = 0, 0

    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                matches.append(i - j)
                j = prefix[j - 1]
        else:
            if j != 0:
                j = prefix[j - 1]
            else:
                i += 1

    return matches


def semantic_entropy(
        label_token: torch.Tensor,
        response_sequence: torch.Tensor,
        scores: Tuple[torch.Tensor],
):
    scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)
    scores = torch.nn.functional.softmax(scores, dim=0)
    try:
        label_token_list = label_token.cpu().tolist()
        if label_token_list[-1] == 220:
            label_token = label_token[:-1]
        if label_token_list[0] == 198:
            label_token = label_token[1:]
        seq_idx = kmp_search(response_sequence.cpu().tolist(), label_token.cpu().tolist())
    except IndexError:
        print(response_sequence)
        print(label_token)
        raise ValueError
    if seq_idx:
        seq_idx = torch.arange(start=seq_idx[0],
                               end=seq_idx[0] + label_token.shape[0],
                               device=label_token.device,
                               dtype=torch.long)
    else:
        return -1.0
    scores = scores[label_token, seq_idx]
    scores = torch.where(torch.isnan(scores), torch.full_like(scores, 0.0), scores)
    entropy = -torch.sum(scores * torch.log(scores))
    return entropy.cpu().item()


def semantic_entropy_score_only(
        response_sequence: torch.Tensor,
        scores: Tuple[torch.Tensor],
):
    scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)
    scores = torch.nn.functional.softmax(scores, dim=0)

    scores = scores[response_sequence, torch.arange(response_sequence.shape[0],
                                                    device=response_sequence.device,
                                                    dtype=torch.long)]
    # scores = torch.where(torch.isnan(scores), torch.full_like(scores, 0.0), scores)
    entropy = -torch.sum(scores * torch.log(scores))
    if torch.isnan(entropy):
        print(response_sequence)
        print(scores)
        raise ValueError
    entropy = torch.where(torch.isnan(entropy), torch.full_like(entropy, 0.0), entropy)
    return entropy.cpu().item()


def load_wos_taxnomy(split: int):
    skipped_label = ['Electrical generator', 'electrical generator']
    with open('../dataset/wos/wos.taxnomy', 'r') as f:
        lines = f.readlines()
        lines = [line.split('\t') for line in lines]
        lines = lines[1:]
    round_dict = {
        'n1': [],
        'n2': [],
        'n3': [],
        'n4': []
    }
    if split == 4:
        for line in lines:
            if line[0] in ['computer science']:
                for label in line[1:]:
                    round_dict['n1'].append(clean_label(label))
            elif line[0] in ['electrical and computer engineering']:
                for label in line[1:]:
                    if label not in skipped_label:
                        round_dict['n1'].append(clean_label(label))
            elif line[0] in ['Medical']:
                for label in line[1:]:
                    round_dict['n2'].append(clean_label(label))
            elif line[0] in ['civil engineering', 'Psychology']:
                for label in line[1:]:
                    round_dict['n3'].append(clean_label(label))
            elif line[0] in ['mechanical and aerospace engineering', 'biochemistry']:
                for label in line[1:]:
                    round_dict['n4'].append(clean_label(label))
    elif split == 6:
        # print(lines)
        round_dict['n5'] = []
        round_dict['n6'] = []
        for line in lines:
            if line[0] in ['computer science']:
                for label in line[1:]:
                    round_dict['n1'].append(clean_label(label))
            elif line[0] in ['electrical and computer engineering']:
                for label in line[1:]:
                    if label not in skipped_label:
                        round_dict['n2'].append(clean_label(label))
            elif line[0] in ['Medical']:
                for label in line[1:int((len(line) - 1) / 2)]:
                    round_dict['n3'].append(clean_label(label))
                for label in line[int((len(line) - 1) / 2):]:
                    round_dict['n4'].append(clean_label(label))
            elif line[0] in ['civil engineering', 'Psychology']:
                for label in line[1:]:
                    round_dict['n5'].append(clean_label(label))
            elif line[0] in ['mechanical and aerospace engineering', 'biochemistry']:
                for label in line[1:]:
                    round_dict['n6'].append(clean_label(label))
    elif split == 8:
        round_dict['n7'] = []
        round_dict['n8'] = []
        for line in lines:
            if line[0] in ['computer science']:
                for label in line[1:]:
                    if label not in skipped_label:
                        round_dict['n1'].append(clean_label(label))
            elif line[0] in ['electrical and computer engineering']:
                for label in line[1:]:
                    if label not in skipped_label:
                        round_dict['n2'].append(clean_label(label))
            elif line[0] in ['Medical']:
                for label in line[1:int((len(line) - 1) / 2)]:
                    round_dict['n3'].append(clean_label(label))
                for label in line[int((len(line) - 1) / 2):]:
                    round_dict['n4'].append(clean_label(label))
            elif line[0] in ['civil engineering']:
                for label in line[1:]:
                    round_dict['n5'].append(clean_label(label))
            elif line[0] in ['Psychology']:
                for label in line[1:]:
                    round_dict['n6'].append(clean_label(label))
            elif line[0] in ['biochemistry']:
                for label in line[1:]:
                    round_dict['n7'].append(clean_label(label))
            elif line[0] in ['mechanical and aerospace engineering']:
                for label in line[1:]:
                    round_dict['n8'].append(clean_label(label))
    else:
        raise ValueError('split should be 4, 6 or 8')
    # print(round_dict)
    return round_dict


def clean_label(label_str):
    if label_str.lower() == 'hepatitis c':
        label_str = 'hepatitis'
    label_str = label_str.replace("'s", '')
    label_str = label_str.replace('_', ' ')
    label_str = label_str.replace('-', ' ')
    label_str = re.findall(r'(?u)\b\w\w+\b', label_str)
    label_str = str(' '.join(label_str)).lower()
    return label_str



def return_exp_name(args):
    arg_dict = vars(args)
    exp_str = ''
    for key, values in arg_dict.items():
        if isinstance(values, bool):
            if values:
                exp_str += f'_{key}'
        elif key == 'seed' or key == 'gpu':
            continue
        elif values == 'none':
            continue
        else:
            exp_str += f'_{key}={values}'
    return exp_str
