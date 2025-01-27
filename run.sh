#!/bin/bash
MODEL_PATH=gpt2

# DATASET=data/$1/
phi0=0.001
phi=composition1_2000_200_${phi0}
DATASET=data/${phi}
# WEIGHT_DECAY=$2
WEIGHT_DECAY=0.1
# N_LAYERS=$3
N_LAYERS=8
# GPU=$4
GPU=3

cot=use_cot
# cot=no_use_cot

OUTPUT_DIR=output/${phi0}_${WEIGHT_DECAY}_${N_LAYERS}$cot

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main.py \
CUDA_VISIBLE_DEVICES=$GPU python main.py \
--data_dir $DATASET \
--model_name_or_path ${MODEL_PATH} \
--weight_decay ${WEIGHT_DECAY} \
--output_dir ${OUTPUT_DIR} \
--max_seq_length 10 \
--max_length 10 \
--block_size 10 \
--train_batch_size 512 \
--eval_batch_size 512 \
--learning_rate 1e-4 \
--gradient_accumulation_steps 1 \
--save_step 50000 \
--save_step_dense 40000 \
--max_steps 8000 \
--do_train \
--scheduler constant_schedule_with_warmup \
--fp16 \
--evaluate_during_training \
--predict_during_training \
--add_tokens \
--n_layer ${N_LAYERS} \
--n_head 12 \
--overwrite_output_dir \
--init_weights\
