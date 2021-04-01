import os

use_gpu_num = '0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num
from core import load_file
import re
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations_with_replacement, combinations
import torch
import transformers
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_file_text(file_path):
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
    return text_file.lower()


def preprocess_text(sentence):
    sentence = sentence.strip().replace(" ", "").replace('<s>', '-')
    sentence = sentence.replace("--------------",
                                "-------------|")
    original_sentence = sentence
    new_sentence = []
    for g in it.groupby(list(sentence)):
        g_list = list(g[1])
        if g[0] != '|' and g[0] != '-':
            new_sentence.append(g[0])
        else:
            new_sentence.extend(g_list)
    sentence = ''.join(new_sentence)
    sentence = re.sub('\\b(-)+\\b', '', sentence)
    sentence = re.sub('\\b(-)+\'', '\'', sentence)
    sentence = re.sub('\'(-)+\\b', '\'', sentence)
    sentence = sentence.replace("-", "|")
    while sentence.startswith('|'):
        sentence = sentence[1:]
    return sentence.lower()


def get_no_punctuation_text(sentence):
    sentence = re.sub('\|+', ' ', sentence)
    return sentence


def count_pause(sentence):
    idx = 0
    cnt_list = []
    while idx < len(sentence):
        while idx < len(sentence) and sentence[idx] != '|':
            idx += 1
        cnt = 0
        while idx < len(sentence) and sentence[idx] == '|':
            idx += 1
            cnt += 1
        cnt_list.append(cnt)
    return cnt_list


def process_path(path):
    all_cnt_list = []
    all_text_list = []
    file_list = [name for name in sorted(os.listdir(path)) if name.endswith('.txt')]
    for name in file_list:
        file_path = os.path.join(path, name)
        text = load_file(file_path, level_list=(), punctuation_list=())
        text['text'] = text['text'].lower()
        print(file_path, len(text['text'].split()))
        print(text['text'])
        text['original_text'] = preprocess_text(text['original_text'])
        text['no_punctuation_text'] = get_no_punctuation_text(text['original_text'])
        file_cnt_list = count_pause(text['original_text'])
        # print(file_cnt_list)
        # text['transcript'] = get_file_text(file_path.replace('asr_text', 'transcription').replace('.txt', '.cha'))
        # print(text['transcript'])
        if len(text['text'].split()) > 20:
            all_cnt_list.extend(file_cnt_list)
            all_text_list.append(text)
    return all_cnt_list, all_text_list


def process_path_21_mmse():
    cn_path = 'ADReSSo21/diagnosis/train/asr_text/cn'
    ad_path = 'ADReSSo21/diagnosis/train/asr_text/ad'

    def load_mmse(dir_path):
        labels = {}
        data = pd.read_csv(os.path.join(dir_path, 'adresso-train-mmse-scores.csv'))
        df = pd.DataFrame(data)

        for index, row in df.iterrows():
            # print(row[0], row['mmse'])
            labels[row['adressfname']] = int(row['mmse'])
        print(labels)
        return labels

    mmse_labels = load_mmse('ADReSSo21/diagnosis/train')

    all_text_list = []
    file_list = [os.path.join(cn_path, name) for name in sorted(os.listdir(cn_path)) if name.endswith('.txt')]
    file_list += [os.path.join(ad_path, name) for name in sorted(os.listdir(ad_path)) if name.endswith('.txt')]
    for file_path in file_list:
        text = load_file(file_path, level_list=(), punctuation_list=())
        text['text'] = text['text'].lower()
        print(file_path, len(text['text'].split()))
        print(text['text'])
        text['original_text'] = preprocess_text(text['original_text'])
        text['no_punctuation_text'] = get_no_punctuation_text(text['original_text'])
        file_cnt_list = count_pause(text['original_text'])
        text['name'] = os.path.basename(file_path).split('.')[0]
        text['mmse'] = mmse_labels[text['name']]
        # print(file_cnt_list)
        # text['transcript'] = get_file_text(file_path.replace('asr_text', 'transcription').replace('.txt', '.cha'))
        # print(text['transcript'])
        if len(text['text'].split()) > 20:
            all_text_list.append(text)
    return all_text_list


def plot_histogram(a):
    a = np.array(a)
    a = a[a < 30]
    _ = plt.hist(a, bins=range(0, 31))
    plt.show()


def find_threshold(hc_cnt_list, hc_text_list, ad_cnt_list, ad_text_list, sentence_threshold):
    max_number = np.maximum(np.max(hc_cnt_list), np.max(ad_cnt_list)) + 1
    hc_bin_cnt = np.bincount(hc_cnt_list, minlength=max_number)
    print(hc_bin_cnt)
    ad_bin_cnt = np.bincount(ad_cnt_list, minlength=max_number)
    print(ad_bin_cnt)

    hc_bin_cnt = hc_bin_cnt[:sentence_threshold]
    ad_bin_cnt = ad_bin_cnt[:sentence_threshold]

    level = 1
    comb_list = combinations(range(2, sentence_threshold), level)
    comb_list = list(comb_list)

    max_sum_diff = 0
    best_comb = []
    for comb in comb_list:
        comb = list(comb)
        comb.append(sentence_threshold)
        pre_level = 0
        sum_diff = 0
        for i in range(level + 1):
            sum_diff += np.abs(np.sum(hc_bin_cnt[pre_level:comb[i] + 1] / len(hc_text_list)) -
                               np.sum(ad_bin_cnt[pre_level:comb[i] + 1]) / len(ad_text_list))
            pre_level = comb[i] + 1
        if sum_diff > max_sum_diff:
            best_comb = comb
            max_sum_diff = sum_diff
        # print(max_sum_diff)
    print(best_comb)


def find_entropy_threshold(hc_text_list, ad_text_list):
    model_name = 'bert-base-uncased'
    # model_name = 'nghuyong/ernie-2.0-en'
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name, model_max_length=512)
    model = transformers.BertForPreTraining.from_pretrained(model_name)
    model.to(device)
    model.eval()

    def get_entropy_score(sentence):
        inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
        inputs.to(device)
        token_list = tokenizer.tokenize(sentence)
        predictions = model(**inputs)
        predictions = predictions['prediction_logits']
        label = inputs["input_ids"].squeeze()
        for idx, token in enumerate(token_list[:int(tokenizer.model_max_length - 1)]):
            if token in [',', '.', ';']:
                label[idx + 1] = -100
        # print(label)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(), label).data
        return float(loss)

    min_epoch_loss_entropy = 0.
    best_period_threshold = None
    for period_threshold in range(1, 1000):
        epoch_loss_classification_ad = []
        epoch_loss_classification_hc = []
        for text in ad_text_list:
            sentence = text['original_text']
            sentence = re.sub('\|{' + str(period_threshold) + ',}', '. ', sentence)
            sentence = re.sub('\|+', ' ', sentence)
            epoch_loss_classification_ad.append(get_entropy_score(sentence))
        for text in hc_text_list:
            sentence = text['original_text']
            sentence = re.sub('\|{' + str(period_threshold) + ',}', '. ', sentence)
            sentence = re.sub('\|+', ' ', sentence)
            epoch_loss_classification_hc.append(get_entropy_score(sentence))
        epoch_loss_classification_diff = \
            abs(np.median(epoch_loss_classification_ad) - np.median(epoch_loss_classification_hc))
        if epoch_loss_classification_diff > min_epoch_loss_entropy:
            min_epoch_loss_entropy = epoch_loss_classification_diff
            best_period_threshold = period_threshold
            print(min_epoch_loss_entropy, best_period_threshold)
    print(best_period_threshold)

    return best_period_threshold


def main():
    # full_wave_enhanced_audio_path = 'ADReSS-IS2020-data/train/asr_text/cc'
    # full_wave_enhanced_audio_path = 'ADReSSo21/diagnosis/train/asr_text/cn'
    full_wave_enhanced_audio_path = 'ADReSSo21/progression/train/asr_text/no_decline'
    hc_cnt_list, hc_text_list = process_path(full_wave_enhanced_audio_path)
    # plot_histogram(hc_cnt_list)

    # full_wave_enhanced_audio_path = 'ADReSS-IS2020-data/train/asr_text/cd'
    # full_wave_enhanced_audio_path = 'ADReSSo21/diagnosis/train/asr_text/ad'
    full_wave_enhanced_audio_path = 'ADReSSo21/progression/train/asr_text/decline'
    ad_cnt_list, ad_text_list = process_path(full_wave_enhanced_audio_path)
    # plot_histogram(ad_cnt_list)

    best_period_threshold = find_entropy_threshold(hc_text_list, ad_text_list)

    find_threshold(hc_cnt_list, hc_text_list, ad_cnt_list, ad_text_list, best_period_threshold)


if __name__ == '__main__':
    main()
