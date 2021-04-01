import os

# use_gpu_num = '0'
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num

import torch
from torch.utils.data.dataset import Dataset, ConcatDataset
import numpy as np
from tqdm import tqdm
from torchvision import datasets, models, transforms
import time
import copy
from transformers import AdamW
from torch import nn
import re
import transformers
import torch.nn.functional as F
from torch.optim import RMSprop
import itertools as it
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error
from datetime import datetime
import json
import argparse
import gc

import core
import dataloaders
import model_head

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(args, model, tokenizer, text_datasets, num_epochs=25, class_num=2):
    since = time.time()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args['learning_rate'])
    # optimizer = RMSprop(model.parameters(), lr=0.000001)
    criterion = nn.CrossEntropyLoss()
    criterion_regression = nn.MSELoss()

    def collate_fn(data_list):
        inputs, labels = [], []
        mmse_label = []
        inputs_shift = []
        file_idx_list = []
        length_label_list = []
        for data in data_list:
            inputs.append(data['text'])
            labels.append(data['label'])
            mmse_label.append(data['label_mmse'])
            file_idx_list.append(data['file_idx'])
            if args['model_name'].startswith('pre_train/'):
                length_label, audio_embedding = core.generate_shift_label(
                    data['text'], data['original_text'], data['audio_embedding'], tokenizer, use_length=False)
                inputs_shift.append(torch.FloatTensor(audio_embedding))
                length_label_list.append(torch.FloatTensor(length_label))

        inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        if args['model_name'].startswith('pre_train/'):
            inputs_shift = pad_sequence(inputs_shift, batch_first=True)
            inputs['shifter_embedding'] = inputs_shift
        # inputs['length_label'] = pad_sequence(length_label_list, batch_first=True)
        return {
            'file_idx': file_idx_list,
            'inputs': inputs,
            'label': torch.tensor(labels),
            'label_mmse': torch.FloatTensor(mmse_label),
        }

    dataloader_dict = {'train': torch.utils.data.DataLoader(text_datasets['train'], batch_size=args['batch_size'],
                                                            shuffle=True, num_workers=4, collate_fn=collate_fn),
                       'test': torch.utils.data.DataLoader(text_datasets['test'], batch_size=1,
                                                           shuffle=False, num_workers=4, collate_fn=collate_fn)}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 9999

    stop = False

    report_dict_list = []

    for epoch in range(num_epochs):
        if stop:
            break
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_report_dict = dict()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss_classification = 0.0
            running_loss_regression = 0.0
            pred_list_classification = []
            ground_truth_list_classification = []
            pred_list_regression = []
            ground_truth_list_regression = []

            # Iterate over data.
            for batch_data in dataloader_dict[phase]:
                inputs = batch_data['inputs']
                for k in inputs:
                    inputs[k] = inputs[k].to(device)
                labels = batch_data['label'].to(device)
                mmse_labels = torch.FloatTensor(batch_data['label_mmse']).unsqueeze(1).to(device)
                file_idx = batch_data['file_idx']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs_classification, outputs_regression = outputs[0], outputs[1]
                    _, preds = torch.max(outputs_classification, 1)
                    loss_classification = criterion(outputs_classification, labels)
                    loss_regression = criterion_regression(outputs_regression, mmse_labels)
                    loss_regression = loss_regression / 100

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if args['train_classification']:
                            loss = loss_classification
                        elif args['train_regression']:
                            loss = loss_regression
                        if args['train_classification'] and args['train_regression']:
                            loss = loss_classification + loss_regression

                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss_classification += loss_classification.item() * labels.size(0)
                running_loss_regression += loss_regression.item() * labels.size(0)
                pred_list_classification.extend(preds.tolist())
                ground_truth_list_classification.extend(labels.data.tolist())
                pred_list_regression.extend(outputs_regression.tolist())
                ground_truth_list_regression.extend(mmse_labels.data.tolist())

                if phase == 'test':
                    if preds != labels.data:
                        print(file_idx, F.softmax(outputs_classification, dim=-1).tolist())
            # if phase == 'train':
            #     scheduler.step()

            report_dict = classification_report(y_true=ground_truth_list_classification,
                                                y_pred=pred_list_classification, output_dict=True)
            epoch_loss_classification = running_loss_classification / len(text_datasets[phase])
            epoch_loss_regression = running_loss_regression / len(text_datasets[phase])
            epoch_acc = report_dict['accuracy']
            epoch_rmse = mean_squared_error(y_true=ground_truth_list_regression,
                                            y_pred=pred_list_regression, squared=False)

            report_dict['loss_classification'] = float(epoch_loss_classification)
            report_dict['loss_regression'] = float(epoch_loss_regression)
            report_dict['rmse'] = epoch_rmse

            loss_sum = 0.0
            if args['train_classification']:
                loss_sum += epoch_loss_classification
            if args['train_regression']:
                loss_sum += epoch_loss_regression
            report_dict['loss_sum'] = loss_sum

            epoch_report_dict[phase] = report_dict

            print('{} classification loss: {:.8f} regression loss: {:.8f} acc: {:.4f} rmse: {:.4f}'.format(
                phase, epoch_loss_classification, epoch_loss_regression, epoch_acc, epoch_rmse))

            for c in range(class_num):
                recall = report_dict[str(c)]['recall']
                precision = report_dict[str(c)]['precision']
                f1_score = report_dict[str(c)]['f1-score']
                print('class :{:d} precision: {:.4f} recall: {:.4f} f1: {:.4f}'.format(c, precision, recall, f1_score))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc

            if phase == 'train':
                if loss_sum <= best_loss:
                    best_loss = loss_sum
                    best_model_wts = copy.deepcopy(model.state_dict())

                if epoch_loss_classification < 0.000001:
                    stop = True

        report_dict_list.append(epoch_report_dict)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    with open(args['training_log_save_path'], 'w') as outfile:
        json.dump(report_dict_list, outfile)

    # load best model weights
    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(), args['model_save_path'])
    return model


def train_a_round(args):
    train_path = 'ADReSS-IS2020-data/train'
    test_path = 'ADReSS-IS2020-data/test'
    test_label_path = 'ADReSS-IS2020-data/test/meta_data_test.txt'

    train_21_path = 'ADReSSo21/diagnosis/train'
    train_21_progression_path = 'ADReSSo21/progression/train'

    model, tokenizer = model_head.load_model(args)
    model = model.to(device)

    train_dataset, test_dataset = None, None
    if args['train_dataset'] == 'ADReSSo21-train':
        train_dataset = dataloaders.ADReSSo21TextTrainDataset(
            train_21_path, args['level_list'], args['punctuation_list'],
            filter_min_word_length=args['train_filter_min_word_length'])
    elif args['train_dataset'] == 'ADReSS20-train':
        train_dataset = dataloaders.ADReSSTextTrainDataset(
            train_path, args['level_list'], args['punctuation_list'],
            filter_min_word_length=args['train_filter_min_word_length'])
    elif args['train_dataset'] == 'ADReSS20-train-transcript':
        train_dataset = dataloaders.ADReSSTextTranscriptTrainDataset(
            train_path)
    elif args['train_dataset'] == 'ADReSSo21-progression-train':
        train_dataset = dataloaders.ADReSSo21TextProgressionTrainDataset(
            train_21_progression_path, args['level_list'], args['punctuation_list'],
            filter_min_word_length=args['train_filter_min_word_length'])

    if args['test_dataset'] == 'ADReSS20-train':
        test_dataset = dataloaders.ADReSSTextTrainDataset(
            train_path, args['level_list'], args['punctuation_list'],
            filter_min_word_length=args['test_filter_min_word_length'])
    elif args['test_dataset'] == 'ADReSS20-test':
        test_dataset = dataloaders.ADReSSTextTestDataset(
            test_path, test_label_path, args['level_list'], args['punctuation_list'],
            filter_min_word_length=args['test_filter_min_word_length'])
    elif args['test_dataset'] == 'ADReSS20':
        test_dataset_list = [dataloaders.ADReSSTextTrainDataset(
            train_path, args['level_list'], args['punctuation_list'],
            filter_min_word_length=args['test_filter_min_word_length']),
            dataloaders.ADReSSTextTestDataset(
                test_path, test_label_path, args['level_list'], args['punctuation_list'],
                filter_min_word_length=args['test_filter_min_word_length'])]
        test_dataset = ConcatDataset(test_dataset_list)
    elif args['test_dataset'] == 'ADReSS20-test-transcript':
        test_dataset = dataloaders.ADReSSTextTranscriptTestDataset(
            test_path, test_label_path)
    elif args['test_dataset'] == 'ADReSSo21-progression-train':
        test_dataset = dataloaders.ADReSSo21TextProgressionTrainDataset(
            train_21_progression_path, args['level_list'], args['punctuation_list'],
            filter_min_word_length=args['train_filter_min_word_length'])

    assert train_dataset is not None
    assert test_dataset is not None

    text_datasets = {'train': train_dataset,
                     'test': test_dataset}

    model = train_model(args, model, tokenizer, text_datasets, num_epochs=args['max_epochs'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_description", required=False,
        default=None,
    )
    ap.add_argument(
        "--batch_size", required=False,
        default=8,
    )
    ap.add_argument(
        "--learning_rate", required=False,
        default=0.000001,
    )
    ap.add_argument(
        "--max_epochs", required=False,
        default=2000,
    )
    ap.add_argument(
        "--level_list", required=False,
        default=(83, 123,),
    )
    ap.add_argument(
        "--punctuation_list", required=False,
        default=(',', '.',),
    )
    ap.add_argument(
        "--model_name", required=False,
        default='bert-base-uncased',
    )
    ap.add_argument(
        "--train_dataset", required=False,
        default='ADReSSo21-train',
    )
    ap.add_argument(
        "--test_dataset", required=False,
        default='ADReSS20',
    )
    ap.add_argument(
        "--train_rounds", required=False,
        default=10,
    )
    ap.add_argument(
        "--train_classification", required=False,
        default=True,
    )
    ap.add_argument(
        "--train_regression", required=False,
        default=False,
    )
    ap.add_argument(
        "--train_filter_min_word_length", required=False,
        default=20,
    )
    ap.add_argument(
        "--test_filter_min_word_length", required=False,
        default=0,
    )
    ap.add_argument(
        "--log_dir", required=False, default='log',
        help="Directory to save."
    )

    args = vars(ap.parse_args())
    if args['model_description'] is not None:
        args['log_dir'] = os.path.join(os.path.abspath(args['log_dir']), args['model_description'])
    else:
        args['log_dir'] = os.path.join(os.path.abspath(args['log_dir']), str(datetime.now()).replace(' ', '_'))
    os.makedirs(args['log_dir'])

    model_dir_path = os.path.join(args['log_dir'], 'models')
    os.makedirs(model_dir_path)

    training_log_dir_path = os.path.join(args['log_dir'], 'training_logs')
    os.makedirs(training_log_dir_path)

    print('args:')
    for arg in args:
        print(str(arg) + ': ' + str(args[arg]))

    with open(os.path.join(os.path.abspath(args['log_dir']), 'args.json'), 'w') as f:
        json.dump(args, f, indent=4)

    for round_idx in range(int(args['train_rounds'])):
        args['training_log_save_path'] = os.path.join(
            training_log_dir_path, 'log_round{:03}.json'.format(round_idx + 1))
        args['model_save_path'] = os.path.join(
            model_dir_path, 'model_round{:03}.pth'.format(round_idx + 1))
        train_a_round(args)

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
