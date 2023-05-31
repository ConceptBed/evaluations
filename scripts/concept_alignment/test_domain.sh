domain="cartoon sketch photo art_painting"
gen_datapath="/home/maitreya/Research/APG/learning-to-learn/evaluations/outputs/pacs/textualinversion"
gen_name="textualinversion"

for dm in $domain; do
    python test.py experiment=erm_all_domain name="logits_MAIN_${dm}_${gen_name}" datamodule.test=true datamodule.domain="$dm" datamodule.test_data_dir="$gen_datapath" +ckpt_path="./checkpoints/pacs.ckpt" 
done


for dm in $domain; do
    python test.py experiment=erm_all_domain name="logits_MAIN_${dm}_original" datamodule.test=true datamodule.domain="$dm" +ckpt_path="./checkpoints/pacs.ckpt"
done

python ./extras/get_uncertainty_domain.py --gen_name="logits_MAIN_{}_${gen_name}" --org_name="logits_MAIN_{}_original"