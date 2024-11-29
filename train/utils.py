from dataclasses import dataclass, field
from typing import Optional,Union
import datasets
import transformers
import deepspeed
import logging
import os
import json

from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset,load_from_disk, concatenate_datasets
LOGGING_LEVEL = logging.WARNING
TB_WRITER_CB: Optional[TensorBoardCallback] = None


@dataclass
class ModelArguments:
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
    no_ptm: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, use NoPTM model (only for fine-tuning)."},
    )
    freeze_flow_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze flow encoders"},
    )
    freeze_burst_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze burst encoders"},
    )
    freeze_embeddings: bool = field(
        default=False,
        metadata={"help": "Freeze embeddings"},
    )


@dataclass
class CommonDataTrainingArguments:
    train_dir: Optional[str] = field(
        metadata={"help": "Directory with training data (Apache Arrow files)"})
    test_dir: Optional[str] = field(default=None, metadata={
        "help": "Directory with testing data (Apache Arrow files)"})
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
    validation_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory with optional input evaluation data to evaluate the perplexity on (Apache Arrow files)"},
    )
    validation_split_percentage: Optional[int] = field(
        default=30,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"}
    )
    data_cache_dir: Optional[str] = field(
        default="/tmp",
        metadata={"help": "Where to store the dataset cache."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_bursts: int = field(
        default=12,
        metadata={
            "help": "The maximum number of sentences after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    max_seq_length: Optional[int] = field(
        default=1296 + 12,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
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


def freeze(model, model_args):
    for name, param in model.base_transformer.named_parameters():
        if model_args.freeze_flow_encoder and (
                "flow_encoder" in name or ("encoder" in name and "position_embeddings" in name)):
            param.requires_grad = False
        if model_args.freeze_burst_encoder and "burst_encoder" in name:
            param.requires_grad = False
        if model_args.freeze_embeddings and (name.startswith("embed") or name.startswith("seg_embed")):
            param.requires_grad = False
    return model


def get_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(LOGGING_LEVEL)
    datasets.utils.logging.set_verbosity(LOGGING_LEVEL)
    transformers.utils.logging.set_verbosity(LOGGING_LEVEL)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return logger


def verify_checkpoint(logger, training_args):
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overwrite it."
            )
        elif (
                last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


def get_90_percent_cpu_count():
    return max(1, int(os.cpu_count() * 0.9))


def load_train_test_datasets(logger, data_args):
    logger.warning("Loading datasets")
    train_split = test_split = None
    if data_args.test_dir is None:
        data_args.test_dir = data_args.train_dir
        train_split = f"train[{data_args.validation_split_percentage}%:]"
        test_split = f"train[:{data_args.validation_split_percentage}%]"

    train_dataset = load_dataset(
        "arrow",
        data_dir=data_args.train_dir,
        split=train_split,
        cache_dir=data_args.data_cache_dir,
    )

    test_dataset = load_dataset(
        "arrow",
        data_dir=data_args.test_dir,
        split=test_split,
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
    train_dataset = train_dataset.add_column("total_bursts", [0] * len(train_dataset))
    test_dataset = test_dataset.add_column("total_bursts", [0] * len(test_dataset))

    return train_dataset, test_dataset

def load_tokenized_dataset(logger,data_args):
    output_dir = "/global/homes/h/haarika/pscratch/network-data-representation/saved_chunked_dataset_11_20"
    datasets_list = []

    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)
        if os.path.isdir(folder_path):
            # Load the dataset from disk
            dataset = load_from_disk(folder_path)
            dataset = dataset.select(range(2000))
            datasets_list.append(dataset)
            break
     
    if datasets_list:
        combined_dataset = concatenate_datasets(datasets_list)
    combined_dataset=combined_dataset.remove_columns(['burst_tokens','rts'])
    train_dataset, test_dataset = combined_dataset.train_test_split(
            train_size=0.8, seed=42
        ).values()
    
    return train_dataset,test_dataset
def initialize_model_with_deepspeed(logger, training_args, get_model, frozen_base=False):
    logger.warning("Initializing deepspeed-optimized model")
    # only if stage 3
    with open(training_args.deepspeed, "r") as f:
        deepspeed_config = json.load(f)

    is_stage_3 = deepspeed_config.get("zero_optimization", {}).get("stage", 0) == 3
    with deepspeed.zero.Init(enabled=False):
        model = get_model()
        if frozen_base:
            for name, param in model.base_transformer.named_parameters():
                param.requires_grad = False
    optimizers = (None, None)
    return model, optimizers


def init_tbwriter(initializer: Union[Trainer, SummaryWriter]) -> None:
    global TB_WRITER_CB

    if isinstance(initializer, SummaryWriter):
        TB_WRITER_CB = lambda: None
        TB_WRITER_CB.tb_writer = initializer

    if "tensorboard" not in initializer.args.report_to:
        return

    if TB_WRITER_CB is not None:
        return

    # hijacking the trainer's Callback with TensorBoardWriter
    for callback in initializer.callback_handler.callbacks:
        if callback.__class__.__name__ == "TensorBoardCallback":
            TB_WRITER_CB = callback
            break


def get_tbwriter() -> SummaryWriter:
    class MagicClass:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                get_logger(__name__).debug("Attempt to write to tensorboard with no SummaryWriter")
                return None
            return method

    if TB_WRITER_CB is None or TB_WRITER_CB.tb_writer is None:
        return MagicClass()
    return TB_WRITER_CB.tb_writer
