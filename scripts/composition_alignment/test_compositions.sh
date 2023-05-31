#!/bin/bash
gen_datapath="/home/maitreya/Research/APG/learning-to-learn/evaluations/outputs/compositions/unet_without_emb_without_multires"
org_datapath="/home/maitreya/Research/APG/learning-to-learn/evaluations/datasets/compositions/compositions_v1_0/vgenome_gt"
gen_name="unet_without_emb_without_multires"
cats="attribute" # relation counting action"

# for cat in $cats; do
#     HYDRA_FULL_ERROR=1 python test.py trainer.gpus=[0] experiment=vqa_json name="logits_VQA_ImageNet_${cat}_${gen_name}" datamodule.categories=[\"$cat\"] datamodule.data_dir="${gen_datapath}" +ckpt_path="" &
# done
# wait


for cat in $cats; do
    HYDRA_FULL_ERROR=1 python test.py trainer.gpus=[0] experiment=vqa_json name="logits_VQA_ImageNet_${cat}_original" datamodule.categories=[\"$cat\"] datamodule.data_dir="$org_datapath" +ckpt_path="" &
done
wait

python ./extras/get_uncertainty_composition.py --gen_name="logits_VQA_ImageNet_{}_${gen_name}" --org_name="logits_VQA_ImageNet_{}_original"