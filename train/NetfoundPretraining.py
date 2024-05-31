import logging
import math
import random
random.seed(42)
import os
from dataclasses import dataclass, field
from typing import Optional
import json

import torch
#import deepspeed
from accelerate.state import AcceleratorState
from datasets import load_dataset, load_metric
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    is_torch_tpu_available,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from typing_extensions import Required

from NetFoundModels import NetfoundDecoderModel
from NetFoundTrainer import NetfoundTrainer
from NetFoundDataCollator import DataCollatorWithMeta
from NetfoundConfig import NetfoundConfig
from NetfoundTokenizer import NetFoundTokenizer

check_min_version("4.15.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
DAEMON_MODE = False


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
    hidden_layers: Optional[int] = field(
        default=12,
        metadata={"help": "Number of hidden layers."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
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
    validation_split_percentage: Optional[int] = field(
        default=30,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=1296 + 12,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    max_bursts: int = field(
        default=12,
        metadata={
            "help": "The maximum number of sentences after tokenization. Sequences longer "
            "than this will be truncated."
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
    max_train_samples: Optional[int] = field(
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
    data_cache_dir: Optional[str] = field(
        #default="/pscratch/sd/s/sguthula/datasets_cache", 
        default="/home/haarika/imp_files/datasets_cache", #haarika's version 
        metadata={"help": "Where to store the dataset cache."},
    )

@record
def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    train_dataset = load_dataset(
        "csv",
        data_files=data_args.train_file,
        split=f"train[{data_args.validation_split_percentage}%:]",
        #split=f"train[:]",
        cache_dir=data_args.data_cache_dir,
    )
    test_dataset = load_dataset(
        "csv",
        data_files=data_args.train_file,
        split=f"train[:{data_args.validation_split_percentage}%]",
        #split=f"train[:]",
        cache_dir=data_args.data_cache_dir,
    )
    if data_args.max_eval_samples is not None:
        test_dataset = test_dataset.select(
            range(min(test_dataset.shape[0], data_args.max_eval_samples))
        )
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(train_dataset.shape[0], data_args.max_train_samples))
        )
    train_dataset = train_dataset.add_column("totalBursts", [0] * len(train_dataset))
    test_dataset = test_dataset.add_column("totalBursts", [0] * len(test_dataset))
    config = NetfoundConfig(
        num_hidden_layers=model_args.hidden_layers,
        no_meta=data_args.no_meta,
        flat=data_args.flat,
    )
    print(f"no_meta{data_args.no_meta}")
    print(f"flat{data_args.flat}")
    config.roformer=False
    config.limit_bursts = data_args.limit_bursts
    print(f"limit bursts: {config.limit_bursts}")
    config.name_or_path = model_args.model_name_or_path
    tokenizer = NetFoundTokenizer(config=config)

    train_dataset = train_dataset.map(tokenizer, batched=True, num_proc=128)
    test_dataset = test_dataset.map(tokenizer, batched=True, num_proc=128)
    data_collator = DataCollatorWithMeta(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        values_clip=65000,
    )

    if model_args.model_name_or_path is not None and os.path.exists(
        model_args.model_name_or_path
    ):
        print(f"Using weights from {model_args.model_name_or_path}")
        get_model = lambda: NetfoundDecoderModel.from_pretrained(
            model_args.model_name_or_path, config=config
        )
    else:
        get_model = lambda: NetfoundDecoderModel(config=config)

    # proper deepspeed init

    if training_args.deepspeed:
        # only if stage 3
        with open(training_args.deepspeed, "r") as f:
            deepspeed_config = json.load(f)

        is_stage_3 = False #deepspeed_config.get("zero_optimization", {}).get("stage", 0) == 3
        with deepspeed.zero.Init(enabled=is_stage_3):
            model = get_model()
        optimizers = (None, None)
    else:
        model = get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.99
        )
        optimizers = (optimizer, lr_scheduler)

    def preprocess_logits_for_metrics(logits, labels):
         #breakpoint()
         if isinstance(logits, tuple):
             return (logits[0].argmax(dim=-1), logits[1])#tuple(i.argmax(dim=-1) for i in logits)
         return (logits.argmax(dim=-1))

    def compute_metrics(eval_preds):
        print("computing metrics")
        import numpy as np
        all_preds, all_labels = eval_preds
        labels = all_labels[0] if type(all_labels) == tuple else all_labels
        preds = all_preds[0] if type(all_preds) == tuple else all_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        ports = all_labels[1] if type(all_labels) == tuple else None
        port_preds = all_preds[1] if type(all_preds) == tuple else None
        proto = all_preds[1] if type(all_preds) == tuple and len(all_preds) > 1 else None
        row, col = np.where((labels != -100))
        token_map = {1: {}, 6: {}, 17: {}}
        for i in range(len(row)):
            curr_proto = proto[row[i]]
            pos_in_burst = col[i] % 109 - 1
            if pos_in_burst < 0:
                continue
            if curr_proto == 6:
                token_id = int(pos_in_burst % 18)
            elif curr_proto == 17:
                token_id = int(pos_in_burst % 12)
            else:
                token_id = int(pos_in_burst % 13)
            if token_id not in token_map[curr_proto]:
                token_map[curr_proto][token_id] = []
            token_map[curr_proto][token_id].append(np.array([labels[row[i]][col[i]], preds[row[i]][col[i]]]))
        for prot in token_map:
            for tok in token_map[prot]:
                lab_preds = np.vstack(token_map[prot][tok])
                token_map[prot][tok] = float(accuracy_score(lab_preds[:, 0], lab_preds[:, 1]))
        #with open('/data/token_preds.json', 'w') as fp:
        with open('/home/haarika/imp_files/token_preds.json', 'w') as fp:
            json.dump(token_map, fp)
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        print("printing metrics")
        return_metrics = {
            "causal_f1": f1_score(labels, preds, average="macro", zero_division=1),
            "causal_prec": precision_score(labels, preds, average="macro", zero_division=1),
            "causal_recall": recall_score(labels, preds, average="macro", zero_division=1),
            "weighted_causal_f1": f1_score(labels, preds, average="weighted", zero_division=1),
            "weighted_causal_prec": precision_score(labels, preds, average="weighted", zero_division=1),
            "weighted_causal_recall": recall_score(labels, preds, average="weighted", zero_division=1),
            "causal_acc": accuracy_score(labels, preds),
        }
        if ports is not None and port_preds is not None:
            return_metrics.update(
                {
                    "port_pred_f1": f1_score(ports, port_preds, average="macro"),
                    "port_pred_prec": precision_score(
                        ports, port_preds, average="macro"
                    ),
                    "port_pred_recall": recall_score(
                        ports, port_preds, average="macro"
                    ),
                    "weighted_port_pred_f1": f1_score(
                        ports, port_preds, average="weighted"
                    ),
                    "weighted_port_pred_prec": precision_score(
                        ports, port_preds, average="weighted"
                    ),
                    "weighted_port_pred_recall": recall_score(
                        ports, port_preds, average="weighted"
                    ),
                    "port_pred_acc": accuracy_score(ports, port_preds),
                }
            )
        return return_metrics

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
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    trainer = NetfoundTrainer(
        #label_names=["ports"],
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        optimizers=optimizers,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
    )
    checkpoint = None
    if training_args.do_train:
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        print(training_args)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # doesn't store tokenizer
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        print(metrics)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if DAEMON_MODE:
    import daemon

    with daemon.DaemonContext():
        main()
else:
    main()

