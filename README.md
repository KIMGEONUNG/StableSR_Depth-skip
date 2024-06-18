<p align="center">
  <img src="https://user-images.githubusercontent.com/22350795/236680126-0b1cdd62-d6fc-4620-b998-75ed6c31bf6f.png" height=40>
</p>

## Depth-skip pruning implementation for StableSR


### Quick start

- Execute doit.sh 
```bash
./doit.sh
```

You can control the depth level as following:

```bash
#!/bin/bash

python scripts/sr_val_ddpm_text_T_vqganfin_depthskip.py \
  --config configs/stableSRNew/depthskip_text_T_512.yaml \
  --ckpt 'ckpts/stablesr_000117.ckpt' \
  --vqgan_ckpt 'ckpts/vqgan_cfw_00011.ckpt' \
  --init-img 'inputs/test_example' \
  --outdir 'outputs' \
  --ddpm_steps 200 \
  --dec_w 0.5 \
  --colorfix_type adain \
  --depth 8 # Control depth level
```
