#!/bin/bash
#SBATCH -A zhenduow
#SBATCH -n 1

#SBATCH --output=job.%J.out
#SBATCH --partition=titan-giant
# And finally run the job​

srun python3 src/train.py -task abs -mode test -test_from models/model_step_148000.pt -batch_size 1 -test_batch_size 1 -beam_size 1 -bert_data_path bert_data/cnndm -log_file logs/val_abs_bert_cnndm -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path logs/abs_bert_cnndm 