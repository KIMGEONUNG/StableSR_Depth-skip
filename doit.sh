#!/bin/bash

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate stablesr

python scripts/sr_val_ddpm_text_T_vqganfin_depthskip.py \
  --config configs/stableSRNew/depthskip_text_T_512.yaml \
  --ckpt 'ckpts/stablesr_000117.ckpt' \
  --vqgan_ckpt 'ckpts/vqgan_cfw_00011.ckpt' \
  --init-img 'inputs/test_example' \
  --outdir 'outputs' \
  --ddpm_steps 200 \
  --dec_w 0.5 \
  --colorfix_type adain \
  --depth 8
