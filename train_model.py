import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

print("Starting process at :", os.getpid())

from ml_translation.dataset import get_datasets, ToolKit
from transformers import Seq2SeqTrainer, EarlyStoppingCallback

from ml_translation.arguments import get_arguments

args = get_arguments()

train, val, _ = get_datasets()

# model_name = "extra_dataset/configs/model" # "Helsinki-NLP/opus-mt-yo-fr"  # google-t5/t5-small
tools = ToolKit(
    args.model_args.pretrained_model_name_or_path, args.model_args.max_length
)

tokenized_train = train.with_transform(tools.preprocess_function)
tokenized_val = val.with_transform(tools.preprocess_function)

print(tools.model)

print(tokenized_train[:5])

trainer = Seq2SeqTrainer(
    model=tools.model,
    args=args.training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tools.tokenizer,
    data_collator=tools.data_collator,
    compute_metrics=tools.compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=args.model_args.early_stopping_patience
        )
    ],
)


print(trainer.evaluate())

trainer.train()

print(trainer.evaluate(metric_key_prefix="test"))
