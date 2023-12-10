#!/bin/sh
model=openai-clip:ViT-B/32

CUDA_VISIBLE_DEVICES=0 python ./model/train.py \
--project 504_neg1 \
--name CocoAndVG \
--model-name=$model \
--train_path /root/code/clip_order/data/train_coco_aug_withneg_adjchange_merge.json \
--test_path /root/code/clip_order/data/visual_genome_attribution_aug.json \
--manualSeed 120 \
--batch_size 64 \
--lr 5e-6 \
--epoch 10 \
--weight_decay 0.1 \
--knowledge_weight 0.2 \
--transformer_layer_num 6 \
--neg_loss_weight 5 \
--device=cuda
