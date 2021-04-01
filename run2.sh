#!/usr/bin/env bash
model_description='bert_base_sequence_level_2-83_123'
CUDA_VISIBLE_DEVICES=0 python3 text_train.py\
 --model_description $model_description >> "out/${model_description}.out"

