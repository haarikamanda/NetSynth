from math import ceil

import torch
from transformers import DataCollatorForLanguageModeling, BatchEncoding
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.basicConfig(filename='/mnt/md0/data/network-data-representation2/src/train/example.log', encoding='utf-8', level=logging.DEBUG)
logger.debug('This message should go to the log file')
field_priority = {
    0: [
        0,
        0.3933541215789797,
        2.724837343196944,
        0.4832308684732291,
        2.8438824786846673,
        1.2984710798489962,
        5.235013308983419,
        10.976387255280429,
        11.045432578384291,
        9.786027295684129,
        9.802312896694096,
        0.00036126955463748926,
        1.225981592766024,
        1.445179430088,
        2.862401334231415,
        2.7450740496296864,
        2.8210014356048245,
        3.047952465179339,
    ],
    1: [
        0,
        0.6722071147171859,
        4.527998107920855,
        0.5309722357810596,
        2.0904554885811057,
        4.528281573964352,
        7.933993039814744,
        7.459545169846418,
        7.749047020428388,
        10.02397015465052,
        10.639314229562498,
        10.657316103744233,
    ],
    2: [
        0,
        0.8211141331923715,
        2.717867785054973,
        0.7648689169745682,
        2.3996483475611847,
        1.5411495772374082,
        1.891531733171775,
        3.143376263094935,
        4.4035206117262975,
        4.518852964093226,
        2.917441246460702,
        3.371685720868757,
        3.938691445460955,
    ],
}

field_priority[0] = torch.from_numpy(
    np.array(field_priority[0]) / sum(field_priority[0])
)
field_priority[1] = torch.from_numpy(
    np.array(field_priority[1]) / sum(field_priority[1])
)
field_priority[2] = torch.from_numpy(
    np.array(field_priority[2]) / sum(field_priority[2])
)


class DataCollatorWithMeta(DataCollatorForLanguageModeling):
    def __init__(self, values_clip: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_clip = values_clip
        self.keys_to_clip = {
            "iat",
            "bytes",
        }

    @staticmethod
    def _clip_values(values: list[Union[int, float]], clip: Union[int, float]) -> list:
        if clip is not None:
            values = np.clip(values, -clip, clip)
        return values

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = {}
        maxBursts = max([int(example["totalBursts"]) for example in examples])
        for i in range(len(examples)):
            inputs = dict((k, v) for k, v in examples[i].items())
            #print(inputs.keys())
            for key in inputs.keys():
                # if key == "replacedAfter":
                #     if "replaced" not in batch:
                #         batch["replaced"] = []
                if key == "labels" or key == "totalBursts" or key == "replacedAfter" or key == "flowDuration":
                    continue
                if key not in batch:
                    if key != "replacedAfter":
                        batch[key] = []
                if key == "ports":
                    batch[key].append(inputs[key] + 1)
                elif key == "proto":
                    batch[key].append(inputs[key])
                # if key == "replacedAfter":
                #     if inputs[key]>0:
                #         #failsafe for a bug
                #         inputs[key] = 1
                #     else:
                #         inputs[key] = 0
                #     batch["replaced"].append(inputs[key])
                elif key in self.keys_to_clip and self.values_clip is not None:
                    batch[key].append(
                        self._clip_values(
                            inputs[key][: maxBursts * self.tokenizer.max_burst_length],
                            self.values_clip,
                        )
                    )
                else:
                    batch[key].append(
                        inputs[key][: maxBursts * self.tokenizer.max_burst_length]
                    )
        for key in batch.keys():
            batch[key] = torch.Tensor(np.array(batch[key]))
            if (
                key == "input_ids"
                or key == "attention_masks"
                or key == "ports"
                or key == "proto"
            ):
                batch[key] = torch.Tensor(batch[key]).to(torch.long)

        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], batch["proto"], special_tokens_mask=None
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return BatchEncoding(batch)

    def torch_mask_tokens(self, input_ids, protos, **kwargs):
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        #probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # probability_matrix = self.maskIgnoreSeqAndAck(
        #     labels, protos, self.mlm_probability
        # )
        # print("input_ids before masking:", input_ids.shape)
        probability_matrix = self.maskLastBurst(labels)
        # print("this is the probability matrix",probability_matrix)
        # print("This is the length of input ids",len(input_ids))
        # probability_matrix = self.maskPacket(labels, protos,0)
        
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # print("these are the masked indices",masked_indices)
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = (
        #     torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # )
        # input_ids[indices_replaced] = self.tokenizer.mask_token

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 1.0 )).bool() & masked_indices #we will mask 100% of the time if chosen 
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = (
        #     torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        #     & masked_indices3
        #     & ~indices_replaced
        # )
        # random_words = torch.randint(
        #     len(self.tokenizer), labels.shape, dtype=torch.long
        # )
        # input_ids[indices_random] = random_words[indices_random]
        # print("input_ids after masking:",input_ids)
        # print("this is the probability matrix after masking",probability_matrix)
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def maskLastBurst(self, labels): #masking every burst 
        probability_matrix = torch.zeros(labels.shape)
        for j in range(labels.shape[0]):
            i = int(labels.shape[1] / 109) - 1
            if i>0:
                probability_matrix[j][i*108:] = 1
        return probability_matrix
    
    def maskBasedOnPos(self, labels, protos, pos, protoNeeded): #masking every burst 
        probability_matrix = torch.zeros(labels.shape)
        for j in range(labels.shape[0]):
            print("this is the first shape",labels.shape[0])
            for i in range(int(labels.shape[1] / 109)):
                print("this is the second shape",labels.shape[1])
                if protos[j] != protoNeeded:
                    continue
                if protos[j] == 6:  # TCP
                    tokensToMask = [1 + pos + x * 18 + i * 109 for x in range(6)]
                elif protos[j] == 17:  # UDP
                    tokensToMask = [1 + pos + x * 12 + i * 109 for x in range(6)]
                elif protos[j] == 1:  # TCP
                    tokensToMask = [1 + pos + x * 13 + i * 109 for x in range(6)]
                probability_matrix[j][tokensToMask] = 1
        return probability_matrix
    def maskIgnoreSeqAndAck(self, labels, protos, mlm_probability):
        probability_matrix = torch.full(labels.shape, mlm_probability)
        for j in range(labels.shape[0]):
            if protos[j] == 0:  # TCP
                for i in range(int(labels.shape[1] / 109)):
                    tokensToMask = self.flatten_2dlist(
                        [
                            [1 + pos + x * 18 + i * 109 for x in range(6)]
                            for pos in range(7, 11)
                        ]
                    )
                    probability_matrix[j][tokensToMask] = 0
            # added code for entropy based masking. But commenting out for now
            # curr_protocol = protos[j].item()
            # probability_matrix[j] *= field_priority[curr_protocol].repeat([int(ceil(probability_matrix[j].shape[-1]/field_priority[curr_protocol].shape[-1]))])[:probability_matrix[j].shape[-1]]*2
        return probability_matrix

    def flatten_2dlist(self, matrix):
        return [item for row in matrix for item in row]

    def maskPacket(self, labels, protos, pos):
        probability_matrix = torch.zeros(labels.shape)
        for j in range(labels.shape[0]):
            # if torch.logical_and(input_ids[j]>0, input_ids[j] != 65537)[:145].sum()%18 != 0:
            #     continue
            for i in range(int(labels.shape[1] / 145)):
                if protos[j] == 0:  # TCP
                    start = pos * 18
                    end = (pos + 1) * 18
                elif protos[j] == 1:  # UDP
                    start = pos * 12
                    end = (pos + 1) * 12

                elif protos[j] == 2:  # TCP
                    start = pos * 13
                    end = (pos + 1) * 13
                probability_matrix[j][1 + start + i * 145 : 1 + end + i * 145] = 0.8
        return probability_matrix


class DataCollatorForFlowClassification:
    label_names: Dict

    def __init__(self, max_burst_length):
        self.max_burst_length = max_burst_length

    def __call__(self, examples):
        import torch

        first = examples[0]
        maxBursts = max([int(example["totalBursts"]) for example in examples])
        for i in range(len(examples)):
            if "stats" in examples[i]:
                examples[i]["stats"] = [
                    float(t) for t in examples[i]["stats"].split(" ")
                ]
        batch = {}
        if "labels" in first and first["labels"] is not None:
            label = (
                first["labels"].item()
                if isinstance(first["labels"], torch.Tensor)
                else first["labels"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["labels"] for f in examples], dtype=dtype)
        if "proto" in first and first["proto"] is not None:
            label = (
                first["proto"].item()
                if isinstance(first["proto"], torch.Tensor)
                else first["proto"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["proto"] = torch.tensor([f["proto"] for f in examples], dtype=dtype)
        if "flowDuration" in first and first["flowDuration"] is not None:
            label = (
                first["flowDuration"].item()
                if isinstance(first["flowDuration"], torch.Tensor)
                else first["flowDuration"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["flowDuration"] = torch.tensor([f["flowDuration"] for f in examples], dtype=dtype)
        for k, v in first.items():
            if (
                k not in ("labels", "label_ids", "totalBursts", "proto", "flowDuration")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack(
                        [f[k][: maxBursts * self.max_burst_length] for f in examples]
                    )
                else:
                    batch[k] = torch.tensor(
                        [f[k][: maxBursts * self.max_burst_length] for f in examples]
                    )
        return batch

class DataCollatorForNATClassification:
    label_names: Dict

    def __init__(self, max_burst_length):
        self.max_burst_length = max_burst_length

    def __call__(self, examples):
        import torch

        first = examples[0]
        maxBursts = max([int(example["totalBursts"]) for example in examples])
        maxBursts = max(max([int(example["totalBursts2"]) for example in examples]), maxBursts)
        batch = {}
        if "labels" in first and first["labels"] is not None:
            label = (
                first["labels"].item()
                if isinstance(first["labels"], torch.Tensor)
                else first["labels"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["labels"] for f in examples], dtype=dtype)
        if "proto" in first and first["proto"] is not None:
            label = (
                first["proto"].item()
                if isinstance(first["labels"], torch.Tensor)
                else first["proto"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["proto"] = torch.tensor([f["proto"] for f in examples], dtype=dtype)

        if "proto2" in first and first["proto2"] is not None:
            label = (
                first["proto2"].item()
                if isinstance(first["labels"], torch.Tensor)
                else first["proto2"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["proto"] = torch.tensor([f["proto"] for f in examples], dtype=dtype)
            batch["proto2"] = torch.tensor([f["proto2"] for f in examples], dtype=dtype)


        for k, v in first.items():
            if (
                k not in ("labels", "label_ids", "totalBursts", "proto", "totalBursts2", "proto2")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack(
                        [f[k][: maxBursts * self.max_burst_length] for f in examples]
                    )
                else:
                    batch[k] = torch.tensor(
                        [f[k][: maxBursts * self.max_burst_length] for f in examples]
                    )
        return batch
