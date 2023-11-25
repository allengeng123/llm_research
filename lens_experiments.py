"""
This file is only used for implementing lens techniques, and interfaces for generate file to call. 
include loading and storing attention score, providing hooks for updating corresponding layers' attention score, and for each token. 
Realizing during experiments all hidden states and attention score should be stored, regardless of being replaced or not. 

"""
import torch
import numpy as np
import transformer_lens
from transformer_lens.hook_points import HookPoint, LensHandle


"""
I just checked with generation running file, and seems like the mechanisms for loading hooks are ready, 
So this file is for defining hooks need to use. 
"""

"""

"""

att_scores_to_replace = None  # must be initialized before generation starts. Once the parameters are provided, should 
# be performed in the post_process_model in cot_generate.py

def create_att_layerwise_hook(
    layer_idx: int
):
    """
    Create a hook for a specific layer's attention score. 
    To make it work, we need a way to store the loaded attention score somewhere first, so 
    it's possible to use external indexing to extract those information. 

    In general: the hook should know which layer it belongs to, and which token index is being generated next. (these two 
    variables can be nonlocal, and should be referred within this method)
    """
    # first define necessary variables
    curr_generate_token_idx = 0
    def att_score_layerwise_hook(
        att_activation: torch.Tensor,
        hook: HookPoint
    ):
        """
        This hook is used for storing attention score for each layer, and each token. 
        """
        nonlocal curr_generate_token_idx
        nonlocal layer_idx 
        global att_scores_to_replace
        # replace the attention activation; realizing by visiting the line calling the hook, this method
        # should RETURN the new attention activation. 
        att_return = att_scores_to_replace[curr_generate_token_idx][layer_idx]
        curr_generate_token_idx += 1
        return att_return
    return att_score_layerwise_hook


def create_att_replace_hook_dict(num_layers: int):
    """
    This method will return a dictionary, where the keys are just simply layer indices, and values
    are the created hooks, by calling create_att_layerwise_hook. 

    TODO: later this method could be modified to support different types of hooks, and different types of
    layer inputs (could be selected layer indices as a list etc.)
    """
    hook_dict = {}
    for i in range(num_layers):
        hook_dict[i] = create_att_layerwise_hook(i)
    return hook_dict    

    
def att_score_replacement_hook(
    att_activation: torch.Tensor,
    hook: HookPoint
):
    pass