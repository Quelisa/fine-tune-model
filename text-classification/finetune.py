import logging
import os
from datasets import load_dataset, load_metric
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          HfArgumentParser, Trainer, TrainingArguments,
                          default_data_collator, AdamW)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The training data file (.txt)."})
    dev_file: Optional[str] = field(
        default=None, metadata={"help": "The validate data file (.txt)."})
    test_file: Optional[str] = field(
        default=None, metadata={"help": "The predict data file (.txt)."})
    num_labels: Optional[str] = field(
        default=None, metadata={"help": "The number of labels."})
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    logger.info("Training/evaluation parameters %s", training_args)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=data_args.num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    datasets = load_dataset('text',
                            data_files={
                                'train': data_args.train_file,
                                'dev': data_args.dev_file,
                                'predict': data_args.test_file,
                            },
                            cache_di="./data/")

    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    def prepare_features(examples):
        texts = []
        labels = []

        for example in examples['text']:
            text, label = example.split('\t')
            texts.append(text)
            labels.append(label)

        feature = tokenizer(texts,
                            padding=padding,
                            truncation=True,
                            max_length=data_args.max_seq_length)
        feature['labels'] = labels
        return feature

    train_dataset = datasets['train'].map(prepare_features, batch=True)
    eval_dataset = datasets['dev'].map(prepare_features, batch=True)
    predict_dataset = datasets['test'].map(prepare_features, batch=True)

    logger.info("**************** Load data done ******************")

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer,
                                                pad_to_multiple_of=8)
    else:
        data_collator = None

    def compute_metrics(eval_preds):
        metric = load_metric('accuracy', 'f1')
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizer=AdamW,
    )

    if training_args.do_train:
        logger.info("**************** Begin to train ******************")
        train_result = trainer.train(model_args.model_name_or_path)
        trainer.save_model()
        output_train_file = os.path.join(training_args.output_dir,
                                         "train_results.txt")

        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json"))

    results = {}
    if training_args.do_eval:
        logger.info("**************** Begin to evalute ******************")
        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir,
                                        "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    if training_args.do_predict:
        logger.info("**************** Begin to predict ******************")
        predictions, labels, metrics = trainer.predict(
            predict_dataset, metric_key_prefix="predict")

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == '__main__':

    main()
