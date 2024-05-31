import copy
import os
import random
from typing import List, Union, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizer, BatchEncoding


class NetFoundTokenizer(PreTrainedTokenizer):
    max_bursts = 11#12
    max_burst_length = 264+1#108 + 1
    CLS_TOKEN = 65537
    PAD_TOKEN = 0
    mask_token = 65538
    vocab_size = 65539
    isNat = False

    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.max_bursts = config.max_bursts
        self.max_burst_length = config.max_burst_length
        self.p = config.p
        self.pretraining = config.pretraining
        self.name_or_path = config.name_or_path
        self.limit_bursts = config.limit_bursts

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, max_bursts={self.max_bursts}, max_burst_length={self.max_burst_length}, p={self.p})"
        )

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        return [self.CLS_TOKEN, self.PAD_TOKEN]

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        return

    def __len__(self):
        return self.vocab_size

    def padBursts(self, ls, maxBurstLength):
        return [
            i[:maxBurstLength] + [self.PAD_TOKEN] * max((maxBurstLength - len(i)), 0)
            for i in ls
        ]

    def flatten_list(self, ls):
        return [item for sublist in ls for item in sublist]

    def padEntireLs(self, ls, maxBursts, tok):
        padBursts = max(maxBursts - len(ls), 0)
        pads = [tok] * len(ls[0]) * (padBursts)
        flow = self.flatten_list(ls[:maxBursts])
        flow += pads
        return flow

    def padFlow(self, ls, maxBursts):
        return self.padEntireLs(ls, maxBursts, self.PAD_TOKEN)

    def padAttnFlow(self, ls, maxBursts):
        return self.padEntireLs(ls, maxBursts, 0)

    def prependToList(self, ls, tok):
        if tok is not None:
            return [[tok] + i for i in ls]
        else:
            return [[i[0]] + i for i in ls]

    def whiteSpaceSplit(self, ls):
        return [i.strip().split(" ") for i in ls]

    def convert_to_tokens(self, bursts, addOne=False):
        return [
            [int(tok, 16) + (1 if addOne else 0) for tok in burst] for burst in bursts
        ]

    def convert_to_attn(self, bursts):
        return [[1] * len(burst) for burst in bursts]

    def padAttnBurst(self, ls, maxBurstLength):
        return [
            i[:maxBurstLength] + [0] * max((maxBurstLength - len(i)), 0) for i in ls
        ]

    def __call__(self, dataset):
        return self.tokenize(dataset)

    def trunc_flow(self, ls, idxs):
        return [".".join(ls[i].split(".")[:idxs[i]])+"." for i in range(len(ls))]

    def tokenize(self, text, **kwargs):
        dataset = text
        if not self.pretraining and "labels" in dataset:
            labels = np.array(dataset["labels"], dtype=int)
            if self.p > 0:
                num_noise_samples = int(self.p * len(labels))
                indices = random.sample(range(0, len(labels) - 1), num_noise_samples)
                noisy_labels = np.random.random_integers(
                    0, 10, size=(num_noise_samples,)
                )
                labels[indices] = noisy_labels
            labels = labels.tolist()
        if self.limit_bursts:
            protos = dataset["proto"]
            protosToLen = {6: 18, 1: 13, 17: 12, 0:18, 2:12}
            bursts_packets = [[len(j.strip().split(" ")) / protosToLen[int(protos[idx])] for j in
                               dataset["text"][idx].split(".")[:-1]] for idx in range(len(dataset["text"]))]
            idx_cutoff = []
            for flow in bursts_packets:
                sumVal = 0
                idx = -1
                for i in range(len(flow)):
                    sumVal+=flow[i]
                    if sumVal>5:
                        idx = max(i, 1)
                        break
                if idx>0:
                    idx_cutoff.append(idx)
                else:
                    idx_cutoff.append(len(flow))
            input_ids, attention_mask = self.tokenize_fields(
                self.trunc_flow(dataset["text"], idx_cutoff), self.CLS_TOKEN, needAttn=True, addOne=True
            )
            # print(input_ids)
            totalBursts = [len(flow.split(".")) - 1 for flow in self.trunc_flow(dataset["text"], idx_cutoff)]
            direction = self.tokenize_fields(self.trunc_flow(dataset["direction"], idx_cutoff), None)
            bytes = self.tokenize_fields(self.trunc_flow(dataset["bytes"], idx_cutoff), None)
            pktCount = self.tokenize_fields(self.trunc_flow(dataset["pktCount"], idx_cutoff), None)
            iat = self.tokenize_fields(self.trunc_flow(dataset["iat"], idx_cutoff), None)
        else:
            input_ids, attention_mask = self.tokenize_fields(
                dataset["text"], self.CLS_TOKEN, needAttn=True, addOne=True
            )
            totalBursts = [len(flow.split(".")) - 1 for flow in dataset["text"]]
            direction = self.tokenize_fields(dataset["direction"], None)
            bytes = self.tokenize_fields(dataset["bytes"], None)
            pktCount = self.tokenize_fields(dataset["pktCount"], None)
            iat = self.tokenize_fields(dataset["iat"], None)

        if self.isNat:
            input_ids2, attention_mask2 = self.tokenize_fields(
                dataset["text2"], self.CLS_TOKEN, needAttn=True, addOne=True
            )
            totalBursts2 = [len(flow.split(".")) - 1 for flow in dataset["text2"]]
            direction2 = self.tokenize_fields(dataset["direction2"], None)
            bytes2 = self.tokenize_fields(dataset["bytes2"], None)
            pktCount2 = self.tokenize_fields(dataset["pktCount2"], None)
            iat2 = self.tokenize_fields(dataset["iat2"], None)

        if not self.pretraining:
            batchDict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "direction": direction,
                "bytes": bytes,
                "pktCount": pktCount,
                "iat": iat,
                "totalBursts": totalBursts,
            }
            if "labels" in dataset:
                batchDict.update({"labels": labels})
            if self.isNat:
                batchDict.update({
                "input_ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "direction2": direction2,
                "bytes2": bytes2,
                "pktCount2": pktCount2,
                "iat2": iat2,
                "totalBursts2": totalBursts2,
            })
        else:
            batchDict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "direction": direction,
                "bytes": bytes,
                "pktCount": pktCount,
                "iat": iat,
                "totalBursts": totalBursts,
            }

        return BatchEncoding(batchDict)

    def tokenize_fields(self, dataset, prependtoken, needAttn=False, addOne=False):
        if needAttn:
            return [
                self.padFlow(
                    self.padBursts(
                        self.prependToList(
                            self.convert_to_tokens(
                                self.whiteSpaceSplit(flow.split(".")[:-1]), addOne
                            ),
                            prependtoken,
                        ),
                        self.max_burst_length,
                    ),
                    self.max_bursts,
                )
                for flow in dataset
            ], [
                self.padAttnFlow(
                    self.padAttnBurst(
                        self.prependToList(
                            self.convert_to_attn(
                                self.whiteSpaceSplit(flow.split(".")[:-1])
                            ),
                            1,
                        ),
                        self.max_burst_length,
                    ),
                    self.max_bursts,
                )
                for flow in dataset
            ]
        else:
            return [
                self.padFlow(
                    self.padBursts(
                        self.prependToList(
                            self.convert_to_tokens(
                                self.whiteSpaceSplit(flow.split(".")[:-1]), addOne
                            ),
                            prependtoken,
                        ),
                        self.max_burst_length,
                    ),
                    self.max_bursts,
                )
                for flow in dataset
            ]
