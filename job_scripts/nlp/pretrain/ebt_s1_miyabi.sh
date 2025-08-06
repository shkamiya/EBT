#!/bin/bash
# ==== PBS directives ===================================================
#PBS -q short-g
#PBS -l select=1:ncpus=16:mem=100gb
#PBS -l walltime=04:00:00
#PBS -N ebt-xxs-bs_256_s1_lr_
#PBS -o logs/
#PBS -e logs/
#PBS -j oe
#PBS -W group_list=gj26
# ======================================================================
# 過去: #PBS -o logs/pbs/nlp/ebt-xxs-bs_256_s1_lr_${PBS_JOBID}-${PBS_ARRAY_INDEX}.log

module purge
module load singularity

# --- ログは $PBS_O_WORKDIR に出る ---
cd $PBS_O_WORKDIR

export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

export WANDB_API_KEY=ac9bc3f259163957d95686abca5fb49df1713b65

export RUN_NAME="ebt-xxs-bs_256_s1_lr_"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"

export HF_HOME=/work/gj26/b20109/datasets

TASK_ID=0

lr=(0.0012)
alpha=(500)
alpha_lr=(1500)

singularity exec --nv \
  --bind $(pwd):/workspace \
  --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/pki/tls/certs/ca-bundle.crt \
  ~/singularity/pytorch_25.01.sif \
  python train_model.py \
    --dataset_dir "/work/gj26/b20109/datasets" \
    --run_name ${RUN_NAME}${lr[${TASK_ID}]} \
    --modality "NLP" \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    \
    --pretokenize_dataset \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    \
    --normalize_initial_condition \
    --ebt_type "time_embed" \
    --denoising_initial_condition "random_noise" \
    --mcmc_step_size_learnable \
    --mcmc_step_size ${alpha[${TASK_ID}]} \
    --mcmc_step_size_lr_multiplier ${alpha_lr[${TASK_ID}]} \
    --mcmc_num_steps 2 \
    \
    --context_length 256 \
    \
    --gpus "-1" \
    \
    --peak_learning_rate ${lr[${TASK_ID}]} \
    --batch_size_per_device 32 \
    --accumulate_grad_batches 2 \
    --gradient_clip_val 1.0 \
    \
    --weight_decay 0.01 \
    --min_lr_scale 10 \
    --max_steps 1000000 \
    --max_scheduling_steps 1000000 \
    --warm_up_steps 10000 \
    \
    --dataset_name "pajama" \
    --num_workers 12 \
    --validation_split_pct 0.0005 \
    --val_check_interval 15000 \
    \
    --wandb_project 'nlp_pretrain' \
    \
    --log_model_archi \
    --log_gradients \
    \
    --set_matmul_precision "medium" \
    --wandb_watch
    # --is_slurm_run