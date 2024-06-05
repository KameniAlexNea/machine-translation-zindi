from transformers import Seq2SeqTrainingArguments, HfArgumentParser
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(
        default="rnn",
        metadata={
            "help": (
                "Model name, can be either `huggingface` model or `rnn` for custom model"
            )
        },
    )
    max_length: int = field(
        default=200,
        metadata={
            "help": (
                "Max length token to tokenize"
            )
        },
    )
    early_stopping_patience: int = field(
        default=8,
        metadata={
            "help": (
                "Max length token to tokenize"
            )
        },
    )

@dataclass
class Arguments:
    model_args: ModelArguments
    training_args: Seq2SeqTrainingArguments

def get_arguments():
    args = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))
    args = args.parse_args_into_dataclasses()
    args = Arguments(*args)
    return args