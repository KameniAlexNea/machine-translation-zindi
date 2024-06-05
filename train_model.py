import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from ml_translation.model import get_datasets, ToolKit
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback


os.environ["TOKENIZERS_PARALLELISM"] = "true"


train, val, _ = get_datasets()

model_name = "Helsinki-NLP/opus-mt-yo-fr"  # google-t5/t5-small
tools = ToolKit(model_name, 300)

tokenized_train = train.with_transform(tools.preprocess_function)
tokenized_val = val.with_transform(tools.preprocess_function)

print(tools.model)

print(tokenized_train[:5])


training_args = Seq2SeqTrainingArguments(
    output_dir="dyula_to_french",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    weight_decay=1e-3,
    save_total_limit=2,
    num_train_epochs=100,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    report_to="wandb",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    save_only_model=True,
    metric_for_best_model="bleu",
    load_best_model_at_end=True,
    dataloader_num_workers=8,
    gradient_accumulation_steps=4,
    auto_find_batch_size=True,
    dataloader_drop_last=True,
    remove_unused_columns=False
)

trainer = Seq2SeqTrainer(
    model=tools.model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tools.tokenizer,
    data_collator=tools.data_collator,
    compute_metrics=tools.compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
)


print(trainer.evaluate())

trainer.train()

print(trainer.evaluate(metric_key_prefix="test"))
