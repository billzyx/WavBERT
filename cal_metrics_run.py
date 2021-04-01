import os

use_gpu_num = '0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num

import numpy as np
from torch.utils.data.dataset import Dataset, ConcatDataset
import json
import torch
from sklearn.metrics import classification_report, mean_squared_error

import model_head
import dataloaders
import core

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

precision_1_list = []
recall_1_list = []
f1_1_list = []

precision_2_list = []
recall_2_list = []
f1_2_list = []

accuracy_list = []
rmse_list = []

train_path = 'ADReSS-IS2020-data/train'
test_path = 'ADReSS-IS2020-data/test'
test_label_path = 'ADReSS-IS2020-data/test/meta_data_test.txt'
train_21_path = 'ADReSSo21/diagnosis/train'

log_dir_path = 'log/bert_base_sequence_level_2-83_123'
filter_min_word_length = 20

model_weights_dir = 'models'
args_json_path = os.path.join(log_dir_path, 'args.json')

with open(args_json_path, 'r') as f:
    args = json.load(f)

all_round_pred_prob_list_classification = []
all_round_pred_list_regression = []

for model_weights_filename in os.listdir(os.path.join(log_dir_path, model_weights_dir)):
    if model_weights_filename.endswith('.pth'):
        model_weights_path = os.path.join(log_dir_path, model_weights_dir, model_weights_filename)
        model, tokenizer = model_head.load_model(args)
        model.load_state_dict(torch.load(model_weights_path))
        model = model.to(device)
        model.eval()

        test_dataset_list = [dataloaders.ADReSSTextTrainDataset(
            train_path, args['level_list'], args['punctuation_list'],
            filter_min_word_length=filter_min_word_length),
            dataloaders.ADReSSTextTestDataset(
                test_path, test_label_path, args['level_list'], args['punctuation_list'],
                filter_min_word_length=filter_min_word_length)]
        test_dataset = ConcatDataset(test_dataset_list)

        pred_list_classification = []
        pred_prob_list_classification = []
        ground_truth_list_classification = []
        pred_list_regression = []
        ground_truth_list_regression = []

        for data in test_dataset:
            inputs = [data['text']]
            inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
            if args['model_name'].startswith('pre_train/'):
                length_label, audio_embedding = core.generate_shift_label(
                    data['text'], data['original_text'], data['audio_embedding'], tokenizer, use_length=False)
                inputs['shifter_embedding'] = torch.tensor([audio_embedding])
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            labels = torch.LongTensor([data['label']]).to(device)
            mmse_labels = torch.FloatTensor([data['label_mmse']]).to(device)

            outputs = model(inputs)
            outputs_classification, outputs_regression = outputs[0], outputs[1]
            _, preds = torch.max(outputs_classification, 1)

            outputs_classification_prob = torch.softmax(outputs_classification, dim=-1)
            pred_prob_list_classification.append(outputs_classification_prob[0].tolist())

            pred_list_classification.extend(preds.tolist())
            ground_truth_list_classification.extend(labels.data.tolist())
            pred_list_regression.extend(outputs_regression.tolist())
            ground_truth_list_regression.extend(mmse_labels.data.tolist())

        round_report_dict = classification_report(y_true=ground_truth_list_classification,
                                                  y_pred=pred_list_classification, output_dict=True)
        round_rmse = mean_squared_error(y_true=ground_truth_list_regression,
                                        y_pred=pred_list_regression, squared=False)
        round_report_dict['rmse'] = round_rmse

        precision_1_list.append(round_report_dict['0']['precision'])
        recall_1_list.append(round_report_dict['0']['recall'])
        f1_1_list.append(round_report_dict['0']['f1-score'])

        precision_2_list.append(round_report_dict['1']['precision'])
        recall_2_list.append(round_report_dict['1']['recall'])
        f1_2_list.append(round_report_dict['1']['f1-score'])

        accuracy_list.append(round_report_dict['accuracy'])
        rmse_list.append(round_report_dict['rmse'])

        all_round_pred_prob_list_classification.append(pred_prob_list_classification)
        all_round_pred_list_regression.append(pred_list_regression)

print(accuracy_list)
print(rmse_list)

precision_1_mean = np.mean(precision_1_list)
recall_1_mean = np.mean(recall_1_list)
f1_1_mean = np.mean(f1_1_list)
precision_2_mean = np.mean(precision_2_list)
recall_2_mean = np.mean(recall_2_list)
f1_2_mean = np.mean(f1_2_list)
accuracy_mean = np.mean(accuracy_list)
rmse_mean = np.mean(rmse_list)

precision_1_std = np.std(precision_1_list)
recall_1_std = np.std(recall_1_list)
f1_1_std = np.std(f1_1_list)
precision_2_std = np.std(precision_2_list)
recall_2_std = np.std(recall_2_list)
f1_2_std = np.std(f1_2_list)
accuracy_std = np.std(accuracy_list)
rmse_std = np.std(rmse_list)

print('precision_1: {:.2f} \pm {:.2f}'.format(precision_1_mean * 100, precision_1_std * 100))
print('recall_1: {:.2f} \pm {:.2f}'.format(recall_1_mean * 100, recall_1_std * 100))
print('f1_1: {:.2f} \pm {:.2f}'.format(f1_1_mean * 100, f1_1_std * 100))
print('precision_2: {:.2f} \pm {:.2f}'.format(precision_2_mean * 100, precision_2_std * 100))
print('recall_2: {:.2f} \pm {:.2f}'.format(recall_2_mean * 100, recall_2_std * 100))
print('f1_2: {:.2f} \pm {:.2f}'.format(f1_2_mean * 100, f1_2_std * 100))
print('accuracy: {:.2f} \pm {:.2f}'.format(accuracy_mean * 100, accuracy_std * 100))
print('rmse: {:.2f} \pm {:.2f}'.format(rmse_mean, rmse_std))

all_round_pred_prob_list_classification = np.array(all_round_pred_prob_list_classification)
all_round_pred_list_regression = np.array(all_round_pred_list_regression)

all_round_pred_prob_list_classification = np.mean(all_round_pred_prob_list_classification, axis=0)
all_round_pred_list_classification = np.argmax(all_round_pred_prob_list_classification, axis=-1)
all_report_dict = classification_report(y_true=ground_truth_list_classification,
                                        y_pred=all_round_pred_list_classification, output_dict=True)

all_round_pred_list_regression = np.mean(all_round_pred_list_regression, axis=0)
all_rmse = mean_squared_error(y_true=ground_truth_list_regression,
                              y_pred=all_round_pred_list_regression, squared=False)
all_report_dict['rmse'] = all_rmse

print()
print('Ensemble in 10 rounds:')
print('precision_1: {:.2f}'.format(all_report_dict['0']['precision'] * 100))
print('recall_1: {:.2f}'.format(all_report_dict['0']['recall'] * 100))
print('f1_1: {:.2f}'.format(all_report_dict['0']['f1-score'] * 100))
print('precision_2: {:.2f}'.format(all_report_dict['1']['precision'] * 100))
print('recall_2: {:.2f}'.format(all_report_dict['1']['recall'] * 100))
print('f1_2: {:.2f}'.format(all_report_dict['1']['f1-score'] * 100))
print('accuracy: {:.2f}'.format(all_report_dict['accuracy'] * 100))
print('rmse: {:.2f}'.format(all_report_dict['rmse']))
