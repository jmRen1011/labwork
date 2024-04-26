deepspeed --include localhost:0,1,2 fuse_main.py \
  --output_dir mistral-fuse-test-load-4epoch-qv-lora-save \
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 8 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --save_steps 100 \
  --save_total_limit 20 \
  --deepspeed configs/deepspeed_config_stage3.json
  # --load_checkpoint True \
  # --checkpoint_path "./mistral-fuse-new-load-4epoch-qv-lora-save/checkpoint-300"

# torchrun --nproc_per_node 4 main.py \
#   --output_dir mixtral-moe-lora-instruct-sharc \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 1 \
#   --learning_rate 1e-4 \
#   --gradient_checkpointing True \
#   --gradient_accumulation_steps 8 \
#   --bf16 True \
#   --tf32 True \
#   --lr_scheduler_type "constant_with_warmup" \
#   --logging_steps 25 \
#   --save_steps 100 \
#   --save_total_limit 3 \
#   --deepspeed configs/deepspeed_config_stage3.json

#   --model_id tiiuae/falcon-180B \
#   --dataset_path dolly-processed \