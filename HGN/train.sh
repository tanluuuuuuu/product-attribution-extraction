export MODEL=roberta-base
epoch=20
lr=5e-5
wis=1qq3qq5qq7
data_type=MAT_COLOR
connect_type=dot-att

CUDA_VISIBLE_DEVICES=0 python run_hgn.py \
    --train_data_dir=data/$data_type/train_80.txt \
    --dev_data_dir=data/$data_type/test_20.txt \
    --test_data_dir=data/$data_type/test_20.txt \
    --bert_model=${MODEL} \
    --task_name=ner \
    --output_dir=./output/roberta_dot_multi_window_v2 \
    --max_seq_length=128 \
    --num_train_epochs ${epoch} \
    --do_train \
    --gpu_id 0 \
    --learning_rate ${lr} \
    --warmup_proportion=0.1 \
    --train_batch_size=4 \
    --use_bilstm \
    --use_multiple_window \
    --windows_list=${wis} \
    --connect_type=${connect_type}
    # --d_model 768

