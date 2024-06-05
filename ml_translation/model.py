import numpy as np

import datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import evaluate
from datasets.features import Translation, Value
from datasets import Features

source_lang = "dyu"
target_lang = "fr"

def load_bible(langi="dyu", langj="fr", task="NLLB"):
    linesi = open(
        f"extra_dataset/{langi}-{langj}.txt/{task}.{langi}-{langj}.{langi}"
    ).readlines()
    linesj = open(
        f"extra_dataset/{langi}-{langj}.txt/{task}.{langi}-{langj}.{langj}"
    ).readlines()
    data = [
        {
            "ID": "bible_" + str(i),
            "translation": {
                source_lang: linei.strip(),
                target_lang: linej.strip(),
            },
        }
        for i, (linei, linej) in enumerate(zip(linesi, linesj))
    ]
    data = datasets.Dataset.from_list(data, features = Features({"translation": Translation([source_lang, target_lang]), "ID": Value("string")}))
    return data


def get_datasets(dyu_only: bool = False):
    dataset: datasets.DatasetDict = datasets.load_dataset("uvci/Koumankan_mt_dyu_fr")
    train, val, test = dataset["train"], dataset["validation"], dataset["test"]
    ls_data: list[datasets.Dataset] = []
    ls_data.append(train)
    bible_train_dyu = load_bible("dyu")
    ls_data.append(bible_train_dyu)
    if not dyu_only:
        bible_train_bam = load_bible("bam", task="QED")
        bible_train_bm = load_bible("bm")
        ls_data.append(bible_train_bam)
        ls_data.append(bible_train_bm)

    bible_train = datasets.concatenate_datasets(ls_data)
    return bible_train, val, test


def postprocess_text(preds: list[str], labels: list[str]):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


class ToolKit:
    def __init__(
        self, checkpoint: str = "google-t5/t5-small", max_length=512, prefix=""
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=checkpoint
        )
        self.metric = evaluate.load("sacrebleu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.max_length = max_length

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.prefix = prefix  # "translate Dyula to French: "

    def preprocess_function(self, examples):
        inputs = [
            self.prefix + example[self.source_lang]
            for example in examples["translation"]
        ]
        targets = [example[self.target_lang] for example in examples["translation"]]
        model_inputs = self.tokenizer(
            inputs, text_target=targets, max_length=self.max_length, truncation=True
        )
        return model_inputs

    def preprocess_data(self, data: datasets.Dataset):
        return data.map(self.preprocess_function, batched=True, batch_size=100_000)

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
