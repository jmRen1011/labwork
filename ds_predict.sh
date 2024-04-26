# deepspeed --include localhost:0,2 main.py \
#   --output_dir mixtral-moe-lora-instruct-1epoch-save \
#   --num_train_epochs 1 \
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

deepspeed --include localhost:1 --master_port 60000 fuse_main.py --output_dir test_predict --test True --fuse True