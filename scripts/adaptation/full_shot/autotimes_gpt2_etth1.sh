export CUDA_VISIBLE_DEVICES=0
model_name=autotimes
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]
label_len=$[$seq_len - $token_len]

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_full_shot \
  --model $model_name \
  --data ETTh1 \
  --seq_len $seq_len \
  --label_len $label_len \
  --token_len $token_len \
  --test_seq_len $seq_len \
  --test_label_len $label_len \
  --test_pred_len $token_len \
  --batch_size 16 \
  --learning_rate 0.002 \
  --itr 1 \
  --train_epochs 10 \
  --use_amp \
  --llm_ckp_dir checkpoints/GPT2/checkpoint.pth \
  --gpu 0 \
  --des 'Gpt2' \
  --cosine \
  --tmax 10 \
  --mlp_hidden_dim 512
