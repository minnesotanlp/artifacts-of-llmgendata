#for dataset in SChem5Labels Sentiment
for dataset in Sentiment
do
    for mode in sorted shuffle frequency data-frequency
    #for mode in data-frequency
    do
    	CUDA_VISIBLE_DEVICES=0 python3 finetune_roberta.py --model_id roberta-base --dataset_name $dataset --filename ../data/intermodel_data.csv --dataset_mode $mode --target_col human_annots --alpha 0
        CUDA_VISIBLE_DEVICES=0 python3 finetune_roberta.py --model_id roberta-base --dataset_name $dataset --filename ../data/intermodel_data.csv --dataset_mode $mode --target_col model_annots --alpha 0
        CUDA_VISIBLE_DEVICES=0 python3 finetune_roberta.py --model_id roberta-base --dataset_name $dataset --filename ../data/intramodel_data.csv --dataset_mode $mode --target_col model_annots --alpha 0 
    done
done
