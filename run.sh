#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 run_mlm.py\
 --output_dir pre_train/pre_train_level_0_word_predictor_l1_no_mask_word_label\
  --do_train --num_train_epochs 100 --dataloader_num_workers=48 --save_steps=5000