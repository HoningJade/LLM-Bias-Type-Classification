from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import os
os.environ["WANDB_SILENT"] = "true"
from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Sequence
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother

from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_fscore_support

from llama.train_llama2_multilabel import preprocess, make_supervised_data_module, SupervisedDataset

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

def cal_roc_auc(all_labels, all_logits):
    num_labels = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc

def int_to_binary_list(number):
    # Convert the integer to binary and remove the '0b' prefix
    binary_string = bin(number)[2:]

    # Pad the binary string with leading zeros to ensure a fixed length of 3
    binary_string = binary_string.zfill(3)

    # Create a list of integers from the binary string
    binary_list = [int(bit) for bit in binary_string]

    return binary_list

def evaluate():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        training_args.output_dir,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.output_dir,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    config.pad_token_id = tokenizer.pad_token_id



    # Load model and tokenizer
    model = transformers.LlamaForSequenceClassification.from_pretrained(
        training_args.output_dir,
        config=config,
        cache_dir=training_args.cache_dir,
    ).to("cuda")
    model.eval()


    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    eval_set = data_module['eval_dataset']


    y_true, y_pred = [], []
    file1 = open(f"{training_args.output_dir}/output_out.txt", "w")

    preds = []
    y_test = []
    with torch.no_grad():
        for i, example in tqdm(enumerate(eval_set)):
            output = model(example["input_ids"].unsqueeze(0).to("cuda"),
                           example["attention_mask"].unsqueeze(0).to("cuda"))
            logits = output["logits"]
            label = example["labels"]
            y_test.append(label)
            softmax_predictions = torch.softmax(logits, dim=1)
            pred = torch.argmax(softmax_predictions, dim=1).cpu().numpy()
            preds.append(pred)

            file1.write(
                str(label) + " |||" + str(pred) + "\n")

    report = classification_report(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    print("===report===:", report)
    print("===accuracy===:", accuracy)
   






if __name__ == "__main__":
    evaluate()
