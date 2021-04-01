import numpy as np
import os
import json

precision_1_list = []
recall_1_list = []
f1_1_list = []

precision_2_list = []
recall_2_list = []
f1_2_list = []

accuracy_list = []
rmse_list = []

max_accuracy = 0.
min_rmse = 99999.

log_dir_path = 'log/bert_base_sequence_level_2-83_123'

training_log_dir = 'training_logs'

print(log_dir_path)
for training_log_filename in sorted(os.listdir(os.path.join(log_dir_path, training_log_dir))):
    if training_log_filename.endswith('.json'):
        log_json_path = os.path.join(log_dir_path, training_log_dir, training_log_filename)
        with open(log_json_path) as f:
            round_data = json.load(f)
        min_loss = 999999.
        choose_epoch_report_dict = None
        for epoch_report_dict in round_data:
            # loss = epoch_report_dict['train']['loss_classification'] + epoch_report_dict['train']['loss_regression']
            loss = epoch_report_dict['train']['loss_sum']
            if loss < min_loss:
                min_loss = loss
                choose_epoch_report_dict = epoch_report_dict
            max_accuracy = max(max_accuracy, epoch_report_dict['test']['accuracy'])
            min_rmse = min(min_rmse, epoch_report_dict['test']['rmse'])

        precision_1_list.append(choose_epoch_report_dict['test']['0']['precision'])
        recall_1_list.append(choose_epoch_report_dict['test']['0']['recall'])
        f1_1_list.append(choose_epoch_report_dict['test']['0']['f1-score'])

        precision_2_list.append(choose_epoch_report_dict['test']['1']['precision'])
        recall_2_list.append(choose_epoch_report_dict['test']['1']['recall'])
        f1_2_list.append(choose_epoch_report_dict['test']['1']['f1-score'])

        accuracy_list.append(choose_epoch_report_dict['test']['accuracy'])
        rmse_list.append(choose_epoch_report_dict['test']['rmse'])

print(accuracy_list)
print('accuracy_list:', np.argmax(accuracy_list) + 1)
print(rmse_list)
print('rmse_list:', np.argmin(rmse_list) + 1)
# assert len(accuracy_list) == 5

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

print('best accuracy: {:.2f}'.format(max_accuracy * 100))

print('rmse: {:.2f} \pm {:.2f}'.format(rmse_mean, rmse_std))
print('best rmse: {:.2f}'.format(min_rmse))
