#!/bin/bash
gen_datapath="/home/maitreya/Research/APG/learning-to-learn/evaluations/outputs/imagenet/unet_without_emb_without_multires"
gen_name="unet_without_emb_without_multires"

HYDRA_FULL_ERROR=1 python test.py experiment=erm_imagenet_w_outliers name="logits_outliers_ImageNet_${gen_name}" datamodule.test_data_dir="$gen_datapath" +ckpt_path="./checkpoints/imagenet.ckpt"
HYDRA_FULL_ERROR=1 python test.py experiment=erm_imagenet_w_outliers name="logits_outliers_ImageNet_original" +ckpt_path="./checkpoints/imagenet.ckpt"


python ./extras/get_uncertainty_objects.py --gen_name="logits_outliers_ImageNet_$gen_name" --org_name="logits_outliers_ImageNet_original"