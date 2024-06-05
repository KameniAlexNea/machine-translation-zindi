nohup python train_model.py \
    --pretrained_model_name_or_path extra_dataset/configs/model \
    --output_dir dyula_to_french \
    --eval_strategy epoch \
    --save_strategy epoch \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --weight_decay 1e-4 \
    --save_total_limit 2 \
    --num_train_epochs 500 \
    --predict_with_generate \
    --fp16 \
    --push_to_hub false \
    --report_to wandb \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --save_only_model \
    --metric_for_best_model bleu \
    --load_best_model_at_end \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps 4 \
    --auto_find_batch_size \
    --dataloader_drop_last \
    --remove_unused_columns false \
    --early_stopping_patience 12 \
    &> nohup.out &

