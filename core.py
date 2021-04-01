import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import itertools as it
import re
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import warnings
from transformers import BertLayer, BertModel, BertForMaskedLM, BertForPreTraining
from transformers.models.bert.modeling_bert import (
    BertForPreTrainingOutput,
    BertPreTrainingHeads,
    BertEncoder,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers.file_utils import ModelOutput
import torch.nn.functional as F


def load_file(file_path, level_list=(13, 26,), punctuation_list=(',', '.',)):
    text_file = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        sentence = lines[0]
        original_sentence, sentence = preprocess_text(sentence, level_list=level_list,
                                                      punctuation_list=punctuation_list)
        text_file['text'] = sentence
        text_file['original_text'] = original_sentence
    return text_file


def preprocess_text(sentence, level_list=(5, 10, 30), punctuation_list=(',', '.', '...')):
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

    # remove blank token in a word
    sentence = re.sub('\\b(-)+\\b', '', sentence)

    # deal with "'" token
    sentence = re.sub('\\b(-)+\'', '\'', sentence)
    sentence = re.sub('\'(-)+\\b', '\'', sentence)

    sentence = sentence.replace("-", "|")
    while sentence.startswith('|'):
        sentence = sentence[1:]
    # sentence = re.sub('\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|'
    #                   '\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|+', '.... ', sentence)
    # sentence = re.sub('\|{30,}', '... ', sentence)
    assert len(level_list) == len(punctuation_list), 'level_list and punctuation_list must have the same length.'
    for level, punctuation in zip(reversed(level_list), reversed(punctuation_list)):
        sentence = re.sub('\|{' + str(level) + ',}', punctuation + ' ', sentence)
    sentence = re.sub('\|+', ' ', sentence)
    return original_sentence, sentence


def token_match(sentence, original_sentence, tokenizer):
    token_list = tokenizer.tokenize(sentence)
    original_sentence = original_sentence.lower()
    labels = [0]
    label_span = []
    cursor = 0
    for token in token_list:
        if token.startswith('['):
            continue
        cursor_start = cursor
        for c in token:
            if c.isalpha():
                while cursor < len(original_sentence) and original_sentence[cursor] != c:
                    cursor += 1
                while cursor < len(original_sentence) and original_sentence[cursor] == c:
                    cursor += 1
        temp_cursor = cursor
        # while temp_cursor < len(original_sentence) and not original_sentence[temp_cursor].isalpha():
        #     temp_cursor += 1
        labels.append(temp_cursor - cursor_start)
        label_span.append([cursor_start, temp_cursor])
    labels = np.array(labels, dtype=np.float)
    labels = labels[:int(tokenizer.model_max_length - 1)]
    labels = np.append(labels, 0)
    # labels = np.expand_dims(labels, axis=1)
    labels = labels / 100
    label_span = np.array(label_span)
    label_span = label_span[:int(tokenizer.model_max_length - 2)]
    return labels, label_span


def audio_embedding_match(audio_embedding, label_span):
    audio_embedding_list = []
    audio_embedding_list.append(np.zeros_like(audio_embedding[0]))
    audio_embedding = np.nan_to_num(audio_embedding, nan=0.0)
    for i in range(np.shape(label_span)[0]):
        audio_embedding_frame = np.zeros_like(audio_embedding[0])
        if label_span[i, 0] != label_span[i, 1]:
            audio_embedding_frame = np.mean(audio_embedding[label_span[i, 0]:label_span[i, 1]], axis=0)
        audio_embedding_list.append(audio_embedding_frame)
    audio_embedding_list.append(np.zeros_like(audio_embedding[0]))
    return np.array(audio_embedding_list)


def generate_shift_label(sentence, original_sentence, audio_embedding, tokenizer, use_length=True):
    length_label, shift_label_span = token_match(sentence, original_sentence, tokenizer)
    audio_embedding = audio_embedding_match(audio_embedding, shift_label_span)
    if use_length:
        audio_embedding = np.concatenate([np.expand_dims(length_label, axis=-1), audio_embedding], axis=-1)
    return length_label, audio_embedding


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsAndLengthOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    length_output: Optional[torch.FloatTensor] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsAndWordPredictorLoss(ModelOutput):
    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    word_predictor_loss: Optional[torch.FloatTensor] = None


class BertWordEmbeddingsPredictor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels=1024,
                              out_channels=config.hidden_size, kernel_size=1)
        self.cnn2 = nn.Conv1d(in_channels=config.hidden_size,
                              out_channels=config.hidden_size, kernel_size=1)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, shifter_embedding=None):
        out = shifter_embedding.permute(0, 2, 1)
        out = self.cnn1(out)
        out = out.permute(0, 2, 1)
        out = self.layer_norm(out)
        out = out.permute(0, 2, 1)
        out = self.cnn2(out)
        out = out.permute(0, 2, 1)
        return out


@dataclass
class BertWordEmbeddingsPredictorPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None


class ShiftBertModel(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.config = config
        self.encoder = BertEncoder(config)
        self.word_embedding_predictor = BertWordEmbeddingsPredictor(config)
        if hasattr(self.config, 'word_predictor_pre_training') and not self.config.word_predictor_pre_training:
            for param in self.word_embedding_predictor.parameters():
                param.requires_grad = False
        # self.word_embedding_shifter = BertWordEmbeddingsPredictor(config)
        # self.word_embedding_predictor = nn.Embedding(
        #     config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id,
        #     _weight=self.get_input_embeddings().weight)
        if hasattr(config, 'train_original') and not config.train_original:
            for param in self.embeddings.parameters():
                param.requires_grad = False
            for param in self.pooler.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
        # self.length_predictor = nn.Conv1d(in_channels=config.hidden_size,
        #                      out_channels=1, kernel_size=1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            shifter_embedding=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # use audio embedding if non-zero
        inputs_embeds = self.word_embedding_predictor(shifter_embedding)
        inputs_embeds_original = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.where(torch.sum(torch.abs(shifter_embedding), dim=-1, keepdim=True) > 0.0,
                                    inputs_embeds, inputs_embeds_original)

        if hasattr(self.config, 'word_predictor_pre_training') and self.config.word_predictor_pre_training:
            loss_fct = L1Loss()
            word_predictor_loss = loss_fct(inputs_embeds, inputs_embeds_original.detach())
            word_predictor_loss = word_predictor_loss * 100

        # inputs_embeds_shifter = self.word_embedding_shifter(shifter_embedding)
        # inputs_embeds = inputs_embeds + inputs_embeds_shifter
        # inputs_embeds = torch.where(torch.sum(torch.abs(shifter_embedding), dim=-1, keepdim=True) > 0.0,
        #                             inputs_embeds, inputs_embeds_original)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            # shifter_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # length_output = sequence_output.permute(0, 2, 1)
        # length_output = self.length_predictor(length_output)
        # length_output = length_output.squeeze(1)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        if hasattr(self.config, 'word_predictor_pre_training') and self.config.word_predictor_pre_training:
            return BaseModelOutputWithPoolingAndCrossAttentionsAndWordPredictorLoss(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
                word_predictor_loss=word_predictor_loss,
            )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            # length_output=length_output,
        )


class ShiftBertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)

        self.bert = ShiftBertModel(config)
        for param in self.cls.parameters():
            param.requires_grad = False
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            shifter_embedding=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForPreTraining
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForPreTraining.from_pretrained('bert-base-uncased')

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            shifter_embedding=shifter_embedding,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None

        if hasattr(self.config, 'word_predictor_pre_training') and self.config.word_predictor_pre_training:
            # print(float(total_loss), float(outputs['word_predictor_loss']))
            total_loss = outputs['word_predictor_loss']

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
