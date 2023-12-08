
from dataclasses import dataclass, field
import math
import pathlib
from typing import Dict, Optional, Sequence
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother


from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )



local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    targets=None
) -> Dict:
    conv = get_conversation_template("llama-2")
    # system_message = "Please help me do a fine-grained bias detection task. It is framed as a multi-label classification task, where three binary classifiers predict the presence of each of the three subcategories (i.e., framing, epistemological, and demographic). For each type, 0 represents no bias and 1 represents otherwise. Now, we treat labels such as [0, 0, 1] as binary and map them to decimal. For example, [1, 1, 0] maps to 6. Therefore, you only need to output a decimal number from 0 to 7 to represent your classification. Now please classify the following sentence into a bias type."
    # TODO
    # system_message = "Framing bias is a type of subjective bias that involves the use of one-sided words or phrases containing a particular point of view. Here is an example: 'New York is the greatest state in the northeastern United States.' Does the following sentence have framing bias?"
    # system_message = "Epistemological bias is a type of subjective bias which includes subtle linguistic features that can affect the believability of the texts. Here is an example: People with disabilities are excluded from cultural and social norms that afford pleasure ( sex , sex education , sexual health , marriage ) . Does the following sentence have epistemological bias? "
    system_message = "Demographic bias is a type of subjective bias with word/phrase usage under presuppositions of a particular demographic factor (i.e., gender or religion). Here is an example: Complementarian an alternative Christian view that interprets scripture to teach that women and men as created equal though men are to hold ultimate authority over women in Church and the home . Does the following sentence have demographic bias? "
    conv.set_system_message(system_message)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv.messages = []
        conv.append_message(conv.roles[0], source)
        # print("***conv.get_prompt()", conv.get_prompt())
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids


    assert conv.sep_style == SeparatorStyle.LLAMA2


    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


# def process_label(s):
#     labels = s.split("|")
#     dec = 4 * int(labels[0]) + 2 * int(labels[1]) + 1 * int(labels[2])
#     return dec

def process_label(s):
    labels = s.split("|")
    # print(f"{labels[1]}")
    dec = int(labels[2]) #TODO
    return dec

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, ds_path, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        self.df = pd.read_csv(ds_path, sep="\t\t",
                              engine="python", header=None)
        print(len(self.df))
        sources = self.df[0].tolist()
        self.labels = [process_label(example) for example in self.df[1].tolist()]
        data_dict = preprocess(sources, tokenizer, self.labels)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        print(len(self.input_ids), len(self.labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(self.labels[i])
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset
    rank0_print("Loading data...")

    train_dataset = dataset_cls(data_args.data_path, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_dataset = dataset_cls(data_args.eval_data_path, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    # config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    # config.num_labels = 2

    # Load model and tokenizer
    model = transformers.LlamaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    trainer_save_model_safe(trainer)






if __name__ == "__main__":
    train()
