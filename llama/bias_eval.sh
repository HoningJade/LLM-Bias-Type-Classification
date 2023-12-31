torchrun --nproc_per_node=1 --master_port=20003 llama/evaluate_multilabel.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --bf16 True \
  --data_path data/class_finegrained/train.tsv \
  --output_dir bias_llama_d \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --save_strategy "steps" \
  --save_steps 1200 \
  --save_total_limit 10 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --evaluation_strategy "no" \
  --eval_data_path data/class_finegrained/test.tsv \
