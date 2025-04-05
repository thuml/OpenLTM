export CUDA_VISIBLE_DEVICES=0
model_name=autotimes
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data UnivariateDatasetBenchmark  \
  --load_time_stamp pt \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --d_model 256 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 2 \
  --valid_last \
  --mix_embeds

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data UnivariateDatasetBenchmark  \
  --load_time_stamp pt \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --batch_size 256 \
  --d_model 256 \
  --gpu 0 \
  --use_norm \
  --e_layers 2 \
  --valid_last \
  --mix_embeds \
  --test_dir forecast_ETTh1_autotimes_UnivariateDatasetBenchmark_sl672_it96_ot96_lr0.0005_bt256_wd0_el2_dm256_dff2048_nh8_cosFalse_test_0
done