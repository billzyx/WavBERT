import torch
from torch import nn
import transformers

import core


class Classifier(nn.Module):
    def __init__(self, class_num, text_backbone, text_configuration):
        super(Classifier, self).__init__()

        self.text_backbone = text_backbone
        # self.fc1 = nn.Linear(text_configuration.hidden_size, text_configuration.hidden_size)
        # self.fc2 = nn.Linear(text_configuration.hidden_size, text_configuration.hidden_size)
        self.text_classifier = nn.Linear(text_configuration.hidden_size, class_num)
        self.text_regressionor = nn.Linear(text_configuration.hidden_size, 1)
        self.cnn = nn.Conv1d(in_channels=text_configuration.hidden_size,
                             out_channels=text_configuration.hidden_size, kernel_size=1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        # self.out_fn = nn.LogSoftmax(dim=-1)

    def forward(self, text_features):
        # features: (batch_size, seq_len, feature)
        # labels: (batch_size,), one utterance to one label

        # length_label = text_features['length_label']
        # del text_features['length_label']
        text_outputs = self.text_backbone(**text_features)
        pooled_output = text_outputs[0][:, 1:, :]
        # pooled_output = torch.cat([pooled_output, length_label[:, 1:].unsqueeze(-1)], dim=-1)
        pooled_output = pooled_output.permute(0, 2, 1)
        pooled_output = self.cnn(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = torch.mean(pooled_output, dim=2)
        # pooled_output = self.dropout(pooled_output)
        # pooled_output_1 = self.fc1(pooled_output)
        # pooled_output_1 = self.relu(pooled_output_1)
        # pooled_output_2 = self.fc2(pooled_output)
        # pooled_output_2 = self.relu(pooled_output_2)
        text_logits = self.text_classifier(pooled_output)
        text_regression = self.text_regressionor(pooled_output)
        text_regression = self.relu(text_regression)

        return text_logits, text_regression


def load_model(args):
    model_name = args['model_name']
    configuration = transformers.BertConfig.from_pretrained(model_name)
    configuration.train_original = True
    configuration.word_predictor_pre_training = False
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name, model_max_length=512)
    if model_name.startswith('pre_train/'):
        print('Loading pre-trained model:', model_name)
        text_model = core.ShiftBertModel.from_pretrained(model_name, config=configuration)
    else:
        print('Loading baseline model:', model_name)
        text_model = transformers.BertModel.from_pretrained(model_name, config=configuration)

    model = Classifier(class_num=2, text_backbone=text_model, text_configuration=configuration)
    return model, tokenizer
