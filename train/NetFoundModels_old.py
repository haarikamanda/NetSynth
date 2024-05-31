from typing import Optional, Tuple

import torch.nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
import torch.nn as nn
from transformers.activations import gelu
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.roformer.modeling_roformer import (
    RoFormerAttention,
    RoFormerIntermediate,
    RoFormerOutput,
    RoFormerEmbeddings,
    RoFormerSinusoidalPositionalEmbedding
)
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaIntermediate,
    RobertaOutput,
    RobertaEmbeddings,
)
from transformers.utils import ModelOutput
import copy
from dataclasses import dataclass


def transform_tokens2bursts(hidden_states, num_bursts, max_burst_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(
        hidden_states,
        (hidden_states.size(0), num_bursts, max_burst_length, hidden_states.size(-1)),
    )
    # squash segments into sequence into a single axis (samples * segments, max_segment_length, hidden_size)
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_bursts, max_burst_length, seg_hidden_states.size(-1)
    )

    return hidden_states_reshape


def transform_masks2bursts(hidden_states, num_bursts, max_burst_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(
        hidden_states, (hidden_states.size(0), 1, 1, num_bursts, max_burst_length)
    )
    # squash segments into sequence into a single axis (samples * segments, 1, 1, max_segment_length)
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_bursts, 1, 1, seg_hidden_states.size(-1)
    )

    return hidden_states_reshape


def transform_bursts2tokens(seg_hidden_states, num_bursts, max_burst_length):
    # transform squashed sequence into segments
    hidden_states = seg_hidden_states.contiguous().view(
        seg_hidden_states.size(0) // num_bursts,
        num_bursts,
        max_burst_length,
        seg_hidden_states.size(-1),
    )
    # transform segments into sequence
    hidden_states = hidden_states.contiguous().view(
        hidden_states.size(0), num_bursts * max_burst_length, hidden_states.size(-1)
    )
    return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roformer = config.roformer
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RoFormerAttention(config) if self.roformer else RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = RoFormerIntermediate(config) if self.roformer else RobertaIntermediate(config)
        self.output = RoFormerOutput(config) if self.roformer else RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        seqNo = None
    ):
        if not self.roformer:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )
        else:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                sinusoidal_pos = seqNo,
                output_attentions=output_attentions,
            )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class NetFoundEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.roformer = config.roformer
        self.layer = nn.ModuleList(
            [NetFoundLayer(config) if not self.config.flat else NetFoundLayerFlat(config) for idx in range(config.num_hidden_layers)]
        )
        self.burst_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads
        )
        self.flow_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_bursts + 1, config.hidden_size // config.num_attention_heads
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_bursts=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # output_attentions = True
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_burst_attentions = () if output_attentions else None

        burst_seqs = transform_tokens2bursts(
            hidden_states, num_bursts=num_bursts, max_burst_length=self.config.max_burst_length
        )
        past_key_values_length = 0
        burstSeqNo = self.burst_positions(burst_seqs.shape[:-1], past_key_values_length)[None, None, :, :]
        flow_seqs = transform_bursts2tokens(
            burst_seqs,
            num_bursts=num_bursts,
            max_burst_length=self.config.max_burst_length,
        )[:, :: self.config.max_burst_length]
        flowSeqNo = self.flow_positions(flow_seqs.shape[:-1], past_key_values_length)[None, None, :, :]

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, burstSeqNo, flowSeqNo)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, num_bursts, output_attentions, burstSeqNo, flowSeqNo
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_burst_attentions = all_burst_attentions + (layer_outputs[2],)
            else:
                all_self_attentions = None
                all_burst_attentions = None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_burst_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithFlowAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            flow_attentions=all_burst_attentions,
        )

    def _tie_weights(self):
        original_position_embeddings = None
        for module in self.layer:
            if hasattr(module, "position_embeddings"):
                assert hasattr(module.position_embeddings, "weight")
                if original_position_embeddings is None:
                    original_position_embeddings = module.position_embeddings
                if self.config.torchscript:
                    module.position_embeddings.weight = nn.Parameter(
                        original_position_embeddings.weight.clone()
                    )
                else:
                    module.position_embeddings.weight = (
                        original_position_embeddings.weight
                    )
        return

class NetFoundEmbeddingsWithMeta:
    def __init__(self, config):
        self.metaEmbeddingLayer1 = nn.Linear(config.metaFeatures, 1024)
        self.metaEmbeddingLayer2 = nn.Linear(1024, config.hidden_size)
        self.no_meta = config.no_meta
        self.protoEmbedding = nn.Embedding(65536, config.hidden_size)

    def addMetaEmbeddings(self,
                          embeddings,
                          direction=None,
                          iat=None,
                          bytes=None,
                          pktCount=None,
                          proto=None,):
        linearLayerDtype = self.metaEmbeddingLayer1.weight.dtype
        if not self.no_meta:
            embeddings += self.metaEmbeddingLayer2(
                self.metaEmbeddingLayer1(
                    torch.concat(
                        [
                            direction.unsqueeze(2).to(linearLayerDtype),
                            bytes.unsqueeze(2).to(linearLayerDtype) / 1000,
                            pktCount.unsqueeze(2).to(linearLayerDtype),
                            iat.unsqueeze(2).to(linearLayerDtype) / 1000,
                        ],
                        dim=2,
                    )
                )
            )
        embeddings += (
            self.protoEmbedding(proto).unsqueeze(1).repeat(1, embeddings.shape[1], 1)
        )
        return embeddings

class NetFoundRobertaEmbeddings(RobertaEmbeddings, NetFoundEmbeddingsWithMeta):
    def __init__(self, config):
        RobertaEmbeddings.__init__(self, config)
        NetFoundEmbeddingsWithMeta.__init__(self, config)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        direction=None,
        iat=None,
        bytes=None,
        pktCount=None,
        proto=None,
    ):
        position_ids = self.create_position_ids_from_input_ids(
            input_ids, self.padding_idx, self.position_ids
        )
        input_shape = input_ids.size()
        embeddings = self.word_embeddings(input_ids)
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.addMetaEmbeddings(embeddings, direction, iat, bytes, pktCount, proto)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def create_position_ids_from_input_ids(input_ids, padding_idx, position_ids):
        mask = input_ids.ne(padding_idx).int()
        # position_ids = position_ids.repeat(input_ids.shape[0], 1)*mask
        position_ids = (
            position_ids.repeat(
                input_ids.shape[0], input_ids.shape[1] // position_ids.shape[1]
            )
            * mask
        )
        return position_ids

class NetFoundRoformerEmbeddings(RoFormerEmbeddings, NetFoundEmbeddingsWithMeta):
    def __init__(self, config):
        RoFormerEmbeddings.__init__(self, config)
        NetFoundEmbeddingsWithMeta.__init__(self, config)
        self.roformer = config.roformer

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            direction=None,
            iat=None,
            bytes=None,
            pktCount=None,
            proto=None,
    ):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.addMetaEmbeddings(embeddings, direction, iat, bytes, pktCount, proto)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class NetFoundLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_burst_length = config.max_burst_length
        self.max_bursts = config.max_bursts
        self.hidden_size = config.hidden_size
        self.burst_encoder = TransformerLayer(config)
        self.flow_encoder = TransformerLayer(config)
        self.position_embeddings = nn.Embedding(
            config.max_bursts + 1, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.roformer = config.roformer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_bursts=None,
        output_attentions=False,
        burstSeqNo = None,
        flowSeqNo = None
    ):
        # transform sequences to bursts
        # output_attentions=True
        burst_inputs = transform_tokens2bursts(
            hidden_states, num_bursts=num_bursts, max_burst_length=self.max_burst_length
        )
        burst_masks = transform_masks2bursts(
            attention_mask,
            num_bursts=num_bursts,
            max_burst_length=self.max_burst_length,
        )

        # if self.roformer:
        #     # breakpoint()
        #     past_key_values_length = 0
        #     burstSeqNo = self.sinusoidalPositionEmbed(burst_inputs.shape[:-1], past_key_values_length)[None, None, :, :]
        # else:
        #     burstSeqNo = None

        burst_outputs = self.burst_encoder(
            burst_inputs, burst_masks, output_attentions=output_attentions, seqNo = burstSeqNo
        )

        # flatten bursts back to tokens
        outputs = transform_bursts2tokens(
            burst_outputs[0],
            num_bursts=num_bursts,
            max_burst_length=self.max_burst_length,
        )

        burst_global_tokens = outputs[:, :: self.max_burst_length].clone()
        burst_attention_mask = attention_mask[:, :, :, :: self.max_burst_length].clone()

        burst_positions = torch.arange(1, num_bursts + 1).repeat(outputs.size(0), 1).to(
            outputs.device
        ) * (burst_attention_mask.reshape(-1, num_bursts) >= -1).int().to(
            outputs.device
        )
        outputs[:, :: self.max_burst_length] += self.position_embeddings(
            burst_positions
        )

        # if self.roformer:
        #     # breakpoint()
        #     past_key_values_length = 0
        #     flowSeqNo = self.position_embeddings(burst_global_tokens.shape[:-1], past_key_values_length)[None, None, :, :]
        # else:
        #     flowSeqNo = None

        flow_outputs = self.flow_encoder(
            burst_global_tokens,
            burst_attention_mask,
            output_attentions=output_attentions,
            seqNo = flowSeqNo
        )

        # replace burst representative tokens
        outputs[:, :: self.max_burst_length] = flow_outputs[0]

        return outputs, burst_outputs, flow_outputs


class NetFoundLayerFlat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_burst_length = config.max_burst_length
        self.max_bursts = config.max_bursts
        self.hidden_size = config.hidden_size
        self.burst_encoder = TransformerLayer(config)
        # self.flow_encoder = TransformerLayer(config)
        self.position_embeddings = nn.Embedding(
            config.max_bursts + 1, config.hidden_size, padding_idx=config.pad_token_id
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_bursts=None,
        output_attentions=False,
        burstSeqNo = None,
        flowSeqNo = None
    ):
        burst_inputs = transform_tokens2bursts(
            hidden_states, num_bursts=num_bursts, max_burst_length=self.max_burst_length
        )
        burst_masks = transform_masks2bursts(
            attention_mask,
            num_bursts=num_bursts,
            max_burst_length=self.max_burst_length,
        )
        burst_outputs = self.burst_encoder(
            burst_inputs, burst_masks, output_attentions=output_attentions, seqNo = burstSeqNo
        )
        outputs = transform_bursts2tokens(
            burst_outputs[0],
            num_bursts=num_bursts,
            max_burst_length=self.max_burst_length,
        )
        return outputs, burst_outputs, #, outputs[0]



@dataclass
class BaseModelOutputWithFlowAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    flow_attentions: Optional[Tuple[torch.FloatTensor]] = None


class NetFoundPretrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, NetFoundEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [
                k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore
            ]
            self._keys_to_ignore_on_load_missing = [
                k
                for k in self._keys_to_ignore_on_load_missing
                if k not in del_keys_to_ignore
            ]

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)


class NetFoundBase(NetFoundPretrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if config.roformer:
            self.embeddings = NetFoundRoformerEmbeddings(config)
        else:
            self.embeddings = NetFoundRobertaEmbeddings(config)
        self.seg_embeddings = torch.nn.Embedding(
            num_embeddings=3, embedding_dim=config.hidden_size
        )
        self.encoder = NetFoundEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        direction=None,
        iat=None,
        bytes=None,
        pktCount=None,
        proto=None,
    ):

        embeddings = self.embeddings(
            input_ids, position_ids, direction, iat, bytes, pktCount, proto
        )
        input_shape = input_ids.size()
        device = input_ids.device
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        num_bursts = input_ids.shape[-1] // self.config.max_burst_length
        encoder_outputs = self.encoder(
            embeddings,
            extended_attention_mask,
            num_bursts,
            output_attentions,
            output_hidden_states,
        )
        final_output = encoder_outputs[0]

        if not return_dict:
            return (final_output) + encoder_outputs[1:]

        return BaseModelOutputWithFlowAttentions(
            last_hidden_state=final_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            flow_attentions=encoder_outputs.flow_attentions,
        )

    """
    BaseModel:
    embedding:
        tokens RobertaAttention: vocab->768
        meta:  4->768
    encoder:
    burst : (seqLength+1) X 768
    concat: (seqLength+1)*num_sen X 768

    flow: num_sen X 768
    replace the reps

    """


class LMHead(nn.Module):
    def __init__(self, config):
        config = copy.deepcopy(config)
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class NetFoundLanguageModelling(NetFoundPretrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.base_transformer = NetFoundBase(config)
        self.lm_head = LMHead(config)
        self.attentivePooling = AttentivePooling(config)
        self.portClassifierHiddenLayer = nn.Linear(config.hidden_size, 65536)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(
            input_embeddings, "num_embeddings"
        ):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.base_transformer.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.base_transformer.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        direction=None,
        iat=None,
        bytes=None,
        pktCount=None,
        ports=None,
        proto=None,
    ):
        #output_attentions=True
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.base_transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            direction=direction,
            iat=iat,
            bytes=bytes,
            pktCount=pktCount,
            proto=proto,
        )
        sequence_output = outputs[0]
        #breakpoint()
        prediction_scores = self.lm_head(sequence_output)
        pooled_output = poolingByAttention(
            self.attentivePooling, sequence_output, self.config.max_burst_length
        )
        # pooled_output = poolingByMean(sequence_output, attention_mask, self.config.max_burst_length)
        portLogits = self.portClassifierHiddenLayer(pooled_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )
            # portClassificationLoss = loss_fct(portLogits, torch.min(ports, torch.tensor([1026]).to(ports.device)))
            totalLoss = masked_lm_loss  # + portClassificationLoss
        if not return_dict:
            print(return_dict)
        # ls,ops = MaskedLMOutput(
        #     loss=totalLoss,
        #     logits=prediction_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )


        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((totalLoss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=totalLoss,
            logits=(prediction_scores, proto),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def poolingByConcat(sequence_output, max_burst_length, hidden_size, max_bursts):
    burstReps = sequence_output[:, ::max_burst_length, :].clone()
    pads = torch.zeros(
        burstReps.shape[0],
        hidden_size * (max_bursts - burstReps.shape[1]),
        dtype=burstReps.dtype,
    ).to(burstReps.device)
    return torch.concat(
        [torch.reshape(burstReps, (burstReps.shape[0], -1)), pads], dim=-1
    ).to(burstReps.device)


def poolingByMean(sequence_output, attention_mask, max_burst_length):
    burst_attention = attention_mask[:, ::max_burst_length].detach().clone()
    burstReps = sequence_output[:, ::max_burst_length, :].clone()
    burst_attention = burst_attention / torch.sum(burst_attention, dim=-1).unsqueeze(
        0
    ).transpose(0, 1)
    orig_shape = burstReps.shape
    burstReps = burst_attention.reshape(
        burst_attention.shape[0] * burst_attention.shape[1], -1
    ) * burstReps.reshape((burstReps.shape[0] * burstReps.shape[1], -1))
    return burstReps.reshape(orig_shape).sum(dim=1)


def poolingByAttention(attentivePooling, sequence_output, max_burst_length):
    burstReps = sequence_output[:, ::max_burst_length, :].clone()
    # burstReps = sequence_output
    return attentivePooling(burstReps)


class AttentivePooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_dropout = config.hidden_dropout_prob
        self.lin_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, inputs):
        lin_out = self.lin_proj(inputs)
        attention_weights = torch.tanh(self.v(lin_out)).squeeze(-1)
        attention_weights_normalized = torch.softmax(attention_weights, -1)
        return torch.sum(attention_weights_normalized.unsqueeze(-1) * inputs, 1)


class NetfoundFinetuningModel(NetFoundPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.model_max_length = config.model_max_length
        self.max_burst_length = self.config.max_burst_length
        self.base_transformer = NetFoundBase(config)
        self.attentivePooling = AttentivePooling(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.hiddenLayer = nn.Linear(config.hidden_size, config.hidden_size)
        self.hiddenLayer2 = nn.Linear(config.hidden_size, config.hidden_size)

        # hidden layers for concat
        # self.hiddenLayer = nn.Linear(config.hidden_size*self.config.max_bursts, int(config.hidden_size*self.config.max_bursts/12))
        # self.hiddenLayer2 = nn.Linear(int(config.hidden_size*self.config.max_bursts/12), config.hidden_size)
        # self.classifier = nn.Linear(config.hidden_size + 64, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.attentivePooling = AttentivePooling(config=config)
        self.relu = nn.ReLU()
        # self.softmax=nn.Softmax(dim = -1)

        # Initialize weights and apply final processing
        self.post_init()

    def poolingByAttention(self, sequence_output, max_burst_length):
        burstReps = sequence_output[:, ::max_burst_length, :].clone()
        # burstReps = sequence_output
        return self.attentivePooling(burstReps)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        direction=None,
        iat=None,
        bytes=None,
        pktCount=None,
        proto=None,
        stats=None,
        flowDuration = None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        import numpy as np

        # p = 0.6
        # if not shouldOut:
        #     if (p > 0):
        #         num_noise_samples = int(p * labels.shape[0])
        #         indices = random.sample(range(0, labels.shape[0]), num_noise_samples)
        #         # noisy_labels = torch.from_numpy(np.random.random_integers(0, 7, size=(num_noise_samples,))).to(
        #         #     labels.device)
        #         # noisy_labels = nn.functional.one_hot(noisy_labels, num_classes=labels.shape[1])
        #         # labels[indices] = noisy_labels
        #
        #         input_ids = input_ids[indices]
        #         attention_mask = attention_mask[indices]
        #         token_type_ids = token_type_ids[indices]
        #         direction = direction[indices]
        #         iat = iat[indices]
        #         bytes = bytes[indices]
        #         pktCount = pktCount[indices]
        #         labels = labels[indices]
        if labels is None:
            labels = flowDuration/1000.0
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.base_transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            direction=direction,
            iat=iat,
            bytes=bytes,
            pktCount=pktCount,
            proto=proto,
        )

        sequence_output = outputs[0]
        # detached to keep the pretrained model unchanged
        # sequence_output = sequence_output.detach()
        pooled_output = poolingByAttention(
            self.attentivePooling, sequence_output, self.config.max_burst_length
        )
        # burst_attention = attention_mask[:, ::self.max_burst_length].detach().clone()

        # for pooling revisit and uncomment below
        # burstReps = sequence_output[:,::self.max_burst_length,:].clone()
        # burst_attention = burst_attention/torch.sum(burst_attention,dim = -1).unsqueeze(0).transpose(0,1)
        # orig_shape = burstReps.shape
        # burstReps = burst_attention.reshape(burst_attention.shape[0] * burst_attention.shape[1], -1) * burstReps.reshape((burstReps.shape[0] * burstReps.shape[1], -1))
        # pooled_output = burstReps.reshape(orig_shape).sum(dim = 1)
        # pooled_output = self.dropout(pooled_output)

        # commented the following for concat
        # pooled_output = poolingByConcat(sequence_output, self.max_burst_length, self.config.hidden_size, self.config.max_bursts)
        import numpy as np

        # with open("/data/NetfoundFinetuningRepWithConcat.csv", "a") as f:
        #     np.savetxt(f, np.concatenate(
        #         (pooled_output.detach().cpu().numpy(), labels.unsqueeze(-1).detach().cpu().numpy()), axis=-1))
        # pooled_output = self.poolingByAttention(sequence_output, self.max_burst_length)
        pooled_output = self.hiddenLayer2(self.hiddenLayer(pooled_output))
        if stats is not None:
            logits = self.classifier(torch.concatenate([pooled_output, stats], dim=-1))
        else:
            logits = self.classifier(torch.concatenate([pooled_output], dim=-1))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    logits = self.relu(logits)
                    loss = loss_fct(logits.squeeze(), (labels.squeeze()))
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # errors in processing due to the following. Check
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

class NetfoundDecoderModel(NetFoundPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.model_max_length = config.model_max_length
        self.max_burst_length = self.config.max_burst_length
        self.base_transformer = NetFoundBase(config)
        self.attentivePooling = AttentivePooling(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.hiddenLayer = nn.Linear(config.hidden_size, config.hidden_size)
        self.hiddenLayer2 = nn.Linear(config.hidden_size, config.hidden_size)

        # hidden layers for concat
        # self.hiddenLayer = nn.Linear(config.hidden_size*self.config.max_bursts, int(config.hidden_size*self.config.max_bursts/12))
        # self.hiddenLayer2 = nn.Linear(int(config.hidden_size*self.config.max_bursts/12), config.hidden_size)
        # self.classifier = nn.Linear(config.hidden_size + 64, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.attentivePooling = AttentivePooling(config=config)
        self.relu = nn.ReLU()
        # self.softmax=nn.Softmax(dim = -1)

        # Initialize weights and apply final processing
        self.post_init()

    def poolingByAttention(self, sequence_output, max_burst_length):
        burstReps = sequence_output[:, ::max_burst_length, :].clone()
        # burstReps = sequence_output
        return self.attentivePooling(burstReps)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        direction=None,
        iat=None,
        bytes=None,
        pktCount=None,
        proto=None,
        stats=None,
        flowDuration = None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        import numpy as np

        # p = 0.6
        # if not shouldOut:
        #     if (p > 0):
        #         num_noise_samples = int(p * labels.shape[0])
        #         indices = random.sample(range(0, labels.shape[0]), num_noise_samples)
        #         # noisy_labels = torch.from_numpy(np.random.random_integers(0, 7, size=(num_noise_samples,))).to(
        #         #     labels.device)
        #         # noisy_labels = nn.functional.one_hot(noisy_labels, num_classes=labels.shape[1])
        #         # labels[indices] = noisy_labels
        #
        #         input_ids = input_ids[indices]
        #         attention_mask = attention_mask[indices]
        #         token_type_ids = token_type_ids[indices]
        #         direction = direction[indices]
        #         iat = iat[indices]
        #         bytes = bytes[indices]
        #         pktCount = pktCount[indices]
        #         labels = labels[indices]
        if labels is None:
            labels = flowDuration/1000.0
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.base_transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            direction=direction,
            iat=iat,
            bytes=bytes,
            pktCount=pktCount,
            proto=proto,
        )

        sequence_output = outputs[0]
        # detached to keep the pretrained model unchanged
        sequence_output = sequence_output.detach()
        pooled_output = poolingByAttention(
            self.attentivePooling, sequence_output, self.config.max_burst_length
        )

        
        # burst_attention = attention_mask[:, ::self.max_burst_length].detach().clone()

        # for pooling revisit and uncomment below
        # burstReps = sequence_output[:,::self.max_burst_length,:].clone()
        # burst_attention = burst_attention/torch.sum(burst_attention,dim = -1).unsqueeze(0).transpose(0,1)
        # orig_shape = burstReps.shape
        # burstReps = burst_attention.reshape(burst_attention.shape[0] * burst_attention.shape[1], -1) * burstReps.reshape((burstReps.shape[0] * burstReps.shape[1], -1))
        # pooled_output = burstReps.reshape(orig_shape).sum(dim = 1)
        # pooled_output = self.dropout(pooled_output)

        # commented the following for concat
        # pooled_output = poolingByConcat(sequence_output, self.max_burst_length, self.config.hidden_size, self.config.max_bursts)
        import numpy as np

        # with open("/data/NetfoundFinetuningRepWithConcat.csv", "a") as f:
        #     np.savetxt(f, np.concatenate(
        #         (pooled_output.detach().cpu().numpy(), labels.unsqueeze(-1).detach().cpu().numpy()), axis=-1))
        # pooled_output = self.poolingByAttention(sequence_output, self.max_burst_length)
        pooled_output = self.hiddenLayer2(self.hiddenLayer(pooled_output))
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        # if stats is not None:
        #     logits = self.classifier(torch.concatenate([pooled_output, stats], dim=-1))
        # else:
        #     logits = self.classifier(torch.concatenate([pooled_output], dim=-1))

        # loss = None
        # if labels is not None:
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (
        #             labels.dtype == torch.long or labels.dtype == torch.int
        #         ):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             logits = self.relu(logits)
        #             loss = loss_fct(logits.squeeze(), (labels.squeeze()))
        #         else:
        #             loss = loss_fct(logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels)

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # errors in processing due to the following. Check
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
class NetfoundNATFinetuningModel(NetFoundPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.model_max_length = config.model_max_length
        self.max_burst_length = self.config.max_burst_length
        self.base_transformer = NetFoundBase(config)
        self.attentivePooling = AttentivePooling(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.hiddenLayer = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.hiddenLayer2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.attentivePooling = AttentivePooling(config=config)
        self.relu = nn.ReLU()
        # self.softmax=nn.Softmax(dim = -1)

        # Initialize weights and apply final processing
        self.post_init()

    def poolingByAttention(self, sequence_output, max_burst_length):
        burstReps = sequence_output[:, ::max_burst_length, :].clone()
        # burstReps = sequence_output
        return self.attentivePooling(burstReps)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        direction=None,
        iat=None,
        bytes=None,
        pktCount=None,
        proto=None,
        input_ids2=None,
        attention_mask2=None,
        position_ids2=None,
        output_attentions2=None,
        output_hidden_states2=None,
        direction2=None,
        iat2=None,
        bytes2=None,
        pktCount2=None,
        proto2=None,
        stats=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        import numpy as np

        # p = 0.6
        # if not shouldOut:
        #     if (p > 0):
        #         num_noise_samples = int(p * labels.shape[0])
        #         indices = random.sample(range(0, labels.shape[0]), num_noise_samples)
        #         # noisy_labels = torch.from_numpy(np.random.random_integers(0, 7, size=(num_noise_samples,))).to(
        #         #     labels.device)
        #         # noisy_labels = nn.functional.one_hot(noisy_labels, num_classes=labels.shape[1])
        #         # labels[indices] = noisy_labels
        #
        #         input_ids = input_ids[indices]
        #         attention_mask = attention_mask[indices]
        #         token_type_ids = token_type_ids[indices]
        #         direction = direction[indices]
        #         iat = iat[indices]
        #         bytes = bytes[indices]
        #         pktCount = pktCount[indices]
        #         labels = labels[indices]
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.base_transformer(
            torch.concat([input_ids, input_ids2]),
            attention_mask=torch.concat([attention_mask, attention_mask2]),
            position_ids=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            direction=torch.concat([direction, direction2]),
            iat=torch.concat([iat, iat2]),
            bytes=torch.concat([bytes, bytes2]),
            pktCount=torch.concat([pktCount, pktCount2]),
            proto=torch.concat([proto, proto2])
        )

        sequence_output = outputs[0]
        pooled_output = poolingByAttention(
            self.attentivePooling, sequence_output, self.config.max_burst_length
        )
        pooled_output = torch.concat([pooled_output[:int(pooled_output.shape[0]/2)], pooled_output[int(pooled_output.shape[0]/2):]], dim = -1)
        pooled_output = self.hiddenLayer2(self.hiddenLayer(pooled_output))
        if stats is not None:
            logits = self.classifier(torch.concatenate([pooled_output, stats], dim=-1))
        else:
            logits = self.classifier(torch.concatenate([pooled_output], dim=-1))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    logits = self.relu(logits)
                    loss = loss_fct(
                        logits.squeeze(), (labels.squeeze()).to(torch.float)
                    )
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # errors in processing due to the following. Check
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )



class NetfoundNoPTM(NetFoundPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.model_max_length = config.model_max_length
        self.max_burst_length = self.config.max_burst_length
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.hiddenLayer = nn.Linear(1595, config.hidden_size * 2)
        self.hiddenLayer2 = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.attentivePooling = AttentivePooling(config=config)
        self.relu = nn.ReLU()
        # self.softmax=nn.Softmax(dim = -1)

        # Initialize weights and apply final processing
        self.post_init()

    def poolingByAttention(self, sequence_output, max_burst_length):
        burstReps = sequence_output[:, ::max_burst_length, :].clone()
        # burstReps = sequence_output
        return self.attentivePooling(burstReps)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        direction=None,
        iat=None,
        bytes=None,
        pktCount=None,
        stats=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        input = torch.concatenate(
            [
                input_ids,
                torch.zeros((input_ids.shape[0], 1595 - input_ids.shape[1])).to(
                    input_ids.device
                ),
            ],
            dim=-1,
        )

        pooled_output = self.hiddenLayer2(self.hiddenLayer(input))
        logits = self.classifier(torch.concatenate([pooled_output], dim=-1))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    logits = self.relu(logits)
                    loss = loss_fct(logits.squeeze(), (labels.squeeze()))
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)

        if not return_dict:
            output = (logits,) + pooled_output[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # errors in processing due to the following. Check
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

