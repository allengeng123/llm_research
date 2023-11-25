"""
This file is intended for making convertions for attention heads
"""

import os
import sys
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizerFast
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaConfig, LlamaMLP
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.utils import logging
import numpy as np
import math


tokenizer_path="tokenizer"
tokenizer = LlamaTokenizerFast.from_pretrained(
        tokenizer_path)
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
tokenizer.pad_token = tokenizer.eos_token
EOS = 2
GENERATE_LEN = 100

def load_attention_tensors(attention_path, device):
    """
    Load attention tensor from file
    """
    attention_tensor = torch.load(attention_path, map_location=device)
    attention_tensor_self = torch.load(f"{attention_path[:-3]}_self.pt", map_location=device)
    # need to create a new tuple for including both tensors together. 
    attention_tensor = attention_tensor_self + attention_tensor
    return attention_tensor
    # resulting tensor is a tuple of length generate_length, each element is a tuple of length layer_num, 
    # containing tensors of shape [batch, head_num, query_len, key_len]; 
    # except the first generated token, otherwise query_len is 1. key_len is the length of entire sequence, including both
    # input and generated parts. 
    # unlike visualization implementations, we don't need to conver the tuple into pure numpy array for plotting, but should have a method 
    # compatible for visualization in the future. 
    

def load_sequence(sequence_path, device):
    """
    Load sequence from file; realizing it might not be necessary to decode, since the key is to just set up a correspondance. 
    """
    sequence = torch.load(sequence_path, map_location=device)  
    return sequence  # a tensor with shape [batch, sequence_length]


def token_alignment(longer_sequence, shorter_sequence, save_dir, no_order=False):
    """
    both sequences must be inputs to language model, not outputs of generation. 
    To handle outputs of generation, simply replace longer sequence's output attention with the input's output attention score. 

    Realizing all tokens from shorter_sequence (except those symbols) should all be included in the longer sequence. 
    The alignment will return a dictionary mapping shorter_sequence's tokens to the longer sequence's tokens, by indexes 
    for modifying attention tensors to fit the input. 

    both sequences must be on the same device, and should have same batch size. 

    no order means the tokens in shorter sequence are not necessarily in the same order as in the longer sequence.
    In this case will consider the first occurrence of a token regardless of order following. 
    """
    # First consider the case requiring order. In this case, the shorter sequence's token 
    # should also occur in the same order as in the longer sequence. Thus [1, 2, 3] can only be aligned to ["1", 3, "2", "3"] but not ["1", "3", "2", 3]. 
    # This operation must be performed batch-wise. 
    batch_res = []
    for i in range(longer_sequence.shape[0]):
        # for each batch
        longer_seq = longer_sequence[i]
        shorter_seq = shorter_sequence[i][:-GENERATE_LEN]
        shorter_seq_len = shorter_seq.shape[0]
        
        # first consider the case requiring order. 
        if not no_order:
            # first find the index of each token in the shorter sequence. 
            shorter_seq_token_index = []
            starting_idx = 0
            eos_end = 0
            for token in shorter_seq:
                # identify padding tokens: "2" and skip them. 
                if token == EOS:
                    eos_end += 1
                    continue
                token_idx = (longer_seq[starting_idx:] == token).nonzero(as_tuple=True)[0][:1]
                shorter_seq_token_index.append(token_idx + starting_idx)
                starting_idx = token_idx.item() + 1 + starting_idx
            shorter_seq_token_index = torch.cat(shorter_seq_token_index, dim=0)
            shorter_seq_token_index = torch.stack([torch.arange(eos_end, shorter_seq_len).to(token_idx.device), 
                                               shorter_seq_token_index], dim=0)  # shape [2, token_num]
                
        if no_order:
            # even in this case, repeated tokens should also follow order; e.g, [1, 2, 2, 3] and [2, 1, 2, 3], the first 2s
            # should be aligned, same for the second 2s.
            # Therefore for both lists, need to find the unique tokens first, then create index matching. 
            shorter_seq_unique_tokens = torch.unique(shorter_seq, return_inverse=True)
            longer_seq_unique_tokens = torch.unique(longer_seq, return_inverse=True)
            # now iterate through tokens from shorter sequence
            shorter_tokens = []
            longer_tokens = []
            for idx in range(len(shorter_seq_unique_tokens[0])):
                # first find the index of the token in the longer sequence
                token = shorter_seq_unique_tokens[0][idx]  # this is the token to analyze
                if token == EOS:
                    continue  # skip padding tokens
                token_counts = torch.sum(shorter_seq_unique_tokens[1] == idx).item()  # this is the token occurrence
                token_idxes = (shorter_seq_unique_tokens[1] == idx).nonzero(as_tuple=True)[0]  # this is the token's idx from shorter list
                token_long_inverse = (longer_seq_unique_tokens[0] == token).nonzero(as_tuple=True)[0].item() 
                # this is the token's idx from longer unique list; use it to identify token positions in longer sequence inverse.
                token_long_idxes = (longer_seq_unique_tokens[1] == token_long_inverse).nonzero(as_tuple=True)[0][:token_counts] 
                # now token_idxes and token_long_idxes should form a one-to-one correspondance.
                shorter_tokens.append(token_idxes)
                longer_tokens.append(token_long_idxes)
            shorter_tokens = torch.cat(shorter_tokens, dim=0)
            longer_tokens = torch.cat(longer_tokens, dim=0)  # shape [token_num]
            shorter_seq_token_index = torch.stack([shorter_tokens, longer_tokens], dim=0)  # shape [2, token_num]
        batch_res.append(shorter_seq_token_index)

    torch.save(batch_res, save_dir)
    return batch_res


def attention_tensor_modification(shorter_seq_tensors, longer_seq_tensors, batch_token_alignment):
    """
    Shorter_seq_tensors are tuples of length generated_seq_len, 
    each is also a tuple of shape num_layers, containing tensors of shape [batch, head_num, query_len, key_len].
    token_alignment is a tensor of shape [batch, 2, token_num], where the first row is the shorter sequence's token index, but should 
    be used as a dictionary. 
    
    This method aims for creating a new longer_seq_tensors, by following mechanisms of copying shorter sequence's attention scores
    for corresponding tokens into longer sequences, and set attention score for longer sequence's tokens NOT in shorter sequence to 0. 
    
    generated_seq_len should be consistent for both shorter and longer sequences. 
    """
    longer_seq_input_len = longer_seq_tensors[0][0].shape[-1]  # the key_len. 
    shorter_seq_input_len = shorter_seq_tensors[0][0].shape[-1]
    new_longer_seq_tensors = []
    # the first tensor should pay special attention: only modify the last query, since it affects generation. 
    first_seq = []
    for layer_idx in range(len(longer_seq_tensors[0])):
        layer = longer_seq_tensors[0][layer_idx]
        new_layer = []
        for batch_idx in range(layer.shape[0]):
            to_keep_longer_tensor = layer[batch_idx, :, :-1, :]  # shape [head_num, query_len-1, key_len]; the part don't need to modify
            batch_alignment = batch_token_alignment[batch_idx]  # shape [2, token_num]
            # now copy tokens; 
            to_copy_longer_tensor = torch.zeros([layer.shape[1], longer_seq_input_len]).to(layer.dtype).to(layer.device)  # shape [head_num, key_len]
            """
            Mechanism: for each token from the shorter sequence (represented by the first row of batch_alignment, the index),
            copy the corresponding attention score to the corresponding index in longer sequence (second row of batch_alignment), 
            and remain other locations' attention score to 0.
            """
            shorter_seq_tensors_attention = shorter_seq_tensors[0][layer_idx][batch_idx][:, -1, :]  # shape [head_num, key_len]
            to_copy_longer_tensor[:, batch_alignment[1]] = shorter_seq_tensors_attention[:, batch_alignment[0]]
            # now concatenate the two tensors
            new_layer.append(torch.cat([to_keep_longer_tensor, to_copy_longer_tensor.unsqueeze(1)], dim=1))
        new_layer = torch.stack(new_layer, dim=0)
        first_seq.append(new_layer)
    first_seq = tuple(first_seq)
    new_longer_seq_tensors.append(first_seq)
    del first_seq  # release memory

    for sequence_idx in range(1, len(shorter_seq_tensors)):
        sequences = longer_seq_tensors[sequence_idx]
        new_layers = []
        for layer_idx in range(len(sequences)):  # each layer is a tensor of shape [batch, head_num, query_len, key_len]
            layer = sequences[layer_idx]
            new_layer = []
            for batch_idx in range(layer.shape[0]):
                # for each batch
                # Basically follow the same procedure as handling the first layer, but: generated parts of longer tensor
                # should be all replaced by shorter tensor's attention score.
                # realizing all inputs after the first sequence have query_len being 1 only; 
                new_att_long_tensor = torch.zeros_like(layer[batch_idx])  # shape [head_num, 1, key_len]
                # first map starting index of each token in longer sequence
                new_att_long_tensor[:, :, batch_token_alignment[batch_idx][1]] = \
                        shorter_seq_tensors[sequence_idx][layer_idx][batch_idx][:, :, batch_token_alignment[batch_idx][0]]
                # now copy the output_att score from shorter sequence to longer sequence. 
                new_att_long_tensor[:, :, longer_seq_input_len:] = \
                    shorter_seq_tensors[sequence_idx][layer_idx][batch_idx][:, :, shorter_seq_input_len:]
                new_layer.append(new_att_long_tensor)
            new_layer = torch.stack(new_layer, dim=0)
            new_layers.append(new_layer)
        new_layers = tuple(new_layers)
        new_longer_seq_tensors.append(new_layers)
    new_longer_seq_tensors = tuple(new_longer_seq_tensors)
    return new_longer_seq_tensors


if __name__ == "__main__":
    # for debugging only
    short = load_sequence("outputs_vicuna\\custom\\res\\batch_0.pt", torch.device("cuda"))
    long = load_sequence("outputs_vicuna\\custom\\res\\batch_4.pt", torch.device("cuda"))
    short_tensor = load_attention_tensors("outputs_vicuna\\custom\\attention\\batch_0.pt", torch.device("cuda"))
    long_tensor = load_attention_tensors("outputs_vicuna\\custom\\attention\\batch_4.pt", torch.device("cuda"))
    # batch_token_alignment = token_alignment(long, short, save_dir="token_alignment.pt", no_order=False)
    batch_token_alignment = torch.load("token_alignment.pt", map_location=torch.device("cuda"))
    # new_longer_seq_tensor = attention_tensor_modification(short_tensor, long_tensor, batch_token_alignment)
    # torch.save(new_longer_seq_tensor, "new_longer_seq_tensor.pt")
    new_longer_seq_tensor = torch.load("new_longer_seq_tensor.pt", map_location=torch.device("cuda"))
    print(new_longer_seq_tensor[0][0].shape, new_longer_seq_tensor[-1][0].shape, len(new_longer_seq_tensor))
    a = 3

