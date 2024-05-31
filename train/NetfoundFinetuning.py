# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
import itertools
import logging
import os
import random
import json
import sys
import deepspeed
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import top_k_accuracy_score

import sklearn.metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch

import datasets
import numpy as np
from datasets import load_dataset
from scipy.special import expit

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

from NetFoundDataCollator import DataCollatorForFlowClassification, DataCollatorForNATClassification
from NetFoundModels import NetfoundFinetuningModel, NetfoundNoPTM, NetfoundNATFinetuningModel
from NetFoundTrainer import NetfoundTrainer
from NetfoundConfig import NetfoundConfig
from NetfoundTokenizer import NetFoundTokenizer

check_min_version("4.17.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


DATASET_CACHE_DIR = "/dev/shm/"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    metaFeatures: Optional[int] = field(
        default=4,
        metadata={"help": "number of metadata fields."},
    )
    no_ptm: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, use NoPTM model"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    no_meta: bool = field(
        default=False,
        metadata={"help": "no meta fields"},
    )
    flat: bool = field(
        default=False,
        metadata={"help": "no cross burst encoder"},
    )
    limit_bursts: bool = field(
        default=False,
        metadata={"help": "limit_bursts"},
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "The input testing data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=DATASET_CACHE_DIR,
        metadata={"help": "The directory to store the cached dataset"},
    )
    num_labels: int = field(
        default=None, metadata={"help": "number of classes in the datasets"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=30,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=2904 + 11,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    max_bursts: int = field(
        default=11,
        metadata={
            "help": "The maximum number of sentences after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    p_val: float = field(
        default=0,
        metadata={
            "help": "noise rate"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.30,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    max_train_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

num_classes = 0

def compute_metrics(p: EvalPrediction):
    global num_classes
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_ids = (p.label_ids).astype(int)
    weighted_f1 = f1_score(
        y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
    )
    weighted_prec = precision_score(
        y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
    )
    weighted_recall = recall_score(
        y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_true=label_ids, y_pred=logits.argmax(axis=1))
    print(classification_report(label_ids, logits.argmax(axis=1), digits=5))
    print(confusion_matrix(label_ids, logits.argmax(axis=1)))
    print(top_k_accuracy_score(label_ids, logits, k=10, labels=np.arange(num_classes)))
    return {
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "weighted_prec: ": weighted_prec,
        "weighted_recall": weighted_recall,
    }


isNAT = False

def main():
    global num_classes
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    for i in range(3):
        if data_args.max_train_samples is not None:
            max_train_samples = data_args.max_train_samples
            if max_train_samples > 1:
                train_dataset = load_dataset(
                    "csv",
                    data_files=data_args.train_file,
                    split=f"train[:{int(max_train_samples)}]",
                    cache_dir=data_args.dataset_cache_dir,
                )
            else:
                train_dataset = load_dataset(
                    "csv",
                    data_files=data_args.train_file,
                    split=f"train[:{int(max_train_samples*100)}%]",
                    cache_dir=data_args.dataset_cache_dir,
                )
        else:
            train_dataset = load_dataset(
                "csv", data_files=data_args.train_file, cache_dir=data_args.dataset_cache_dir
            )["train"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = data_args.max_eval_samples
            test_dataset = load_dataset(
                "csv",
                data_files=data_args.test_file,
                split=f"train[:{max_eval_samples}]",
                cache_dir=data_args.dataset_cache_dir,
            )
        else:
            test_dataset = load_dataset(
                "csv", data_files=data_args.test_file, cache_dir=data_args.dataset_cache_dir
            )["train"]
        test_dataset = test_dataset
        train_dataset = train_dataset.add_column(
            "totalBursts", [0] * len(train_dataset)
        )
        test_dataset = test_dataset.add_column("totalBursts", [0] * len(test_dataset))
        config = NetfoundConfig(no_meta=data_args.no_meta, flat=data_args.flat)
        config.pretraining = False
        config.num_labels = data_args.num_labels
        num_classes = data_args.num_labels
        testingTokenizer = NetFoundTokenizer(config=config)
        testingTokenizer.isNat = isNAT
        config.p = data_args.p_val
        config.limit_bursts = data_args.limit_bursts
        print(f"limit bursts: {config.limit_bursts}")
        print(f"noise rate value {config.p}")
        trainingTokenizer = NetFoundTokenizer(config=config)
        trainingTokenizer.isNat = isNAT
        additionalFields = None
        if isNAT:
            additionalFields = {
                "direction2",
                "iat2",
                "bytes2",
                "pktCount2",
                "totalBursts2",
                "ports2",
                "stats2",
                "proto2"}

        train_dataset = train_dataset.map(trainingTokenizer, batched=True, num_proc=40)
        test_dataset = test_dataset.map(testingTokenizer, batched=True, num_proc=40)
        if trainingTokenizer.isNat:
            data_collator = DataCollatorForNATClassification(config.max_burst_length)
        else:
            data_collator = DataCollatorForFlowClassification(config.max_burst_length)
        if model_args.model_name_or_path is not None and os.path.exists(
            model_args.model_name_or_path
        ):
            print(f"Using weights from {model_args.model_name_or_path}")
            if trainingTokenizer.isNat:
                get_model = lambda: NetfoundNATFinetuningModel.from_pretrained(
                    model_args.model_name_or_path, config=config
                )
            else:
                get_model = lambda: NetfoundFinetuningModel.from_pretrained(
                    model_args.model_name_or_path, config=config
                )
        elif model_args.no_ptm:
            get_model = lambda: NetfoundNoPTM(config=config)
        else:
            if trainingTokenizer.isNat:
                get_model = lambda: NetfoundNATFinetuningModel(config=config)
            else:
                get_model = lambda: NetfoundFinetuningModel(config=config)

        # Explicitly freeze parts of model for training
        # for name, param in model.base_transformer.named_parameters():
        #     if "flow_encoder" not in name:
        #         param.requires_grad = False

        # proper deepspeed init
        if training_args.deepspeed:
            # only if stage 3
            with open(training_args.deepspeed, "r") as f:
                deepspeed_config = json.load(f)

            is_stage_3 = deepspeed_config.get("zero_optimization", {}).get("stage", 0) == 3
            with deepspeed.zero.Init(enabled=is_stage_3):
                model = get_model()
            optimizers = (None, None)
        else:
            model = get_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
            if not model_args.no_ptm:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=10000, gamma=0.99
                )
            else:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=100, gamma=0.995
                )
            optimizers = (optimizer, lr_scheduler)

        trainer = NetfoundTrainer(
            model=model,
            extraFields=additionalFields,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=testingTokenizer,
            optimizers=optimizers,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
            data_collator=data_collator,
        )
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # doesn't store tokenizer
            metrics = train_result.metrics

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(eval_dataset=test_dataset)

            # max_eval_samples = (
            #     data_args.max_eval_samples
            #     if data_args.max_eval_samples is not None
            #     else len(test_dataset)
            # )
            # metrics["eval_samples"] = min(max_eval_samples, len(test_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


main()
