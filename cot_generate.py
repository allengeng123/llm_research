import os
import sys


import types
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import transformers
import datasets
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.utils import logging
from dataset_utils import create_dataloader
from hook_methods.head_suppress_hook import *
import numpy as np
import argparse
from attention_align.lens_experiments import *
# from peft import (
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
#     PeftModel,
# )

from typing import Optional, List

logger = logging.get_logger(__name__)
"""
Should follow these steps sequentially: 

Done: 
1. load datasets
2. load model; test with 13B model


To do:
3. set up multi-GPU running
4. evaluation configuration setting
5. passing data in a loop, with defined prompts. 
6. extract output info (need outpus, tokenized outputs, hidden states, and attention map)
    and save those results. 
"""




def acquire_device_dict(num_layers:int):
    """for performing model parallelism on multiple GPUs; 

    Returns: 
    """
    available_gpus = torch.cuda.device_count()
    if available_gpus > 2: 
        available_gpu = available_gpus - 1 # leave the last GPU for data processing
    else:
        available_gpu = available_gpus
    custom_device_map = {}

    num_layers_per_gpu = num_layers // available_gpu

    custom_device_map["model.embed_tokens"] = 0
    custom_device_map["model.norm"] = available_gpu - 1
    custom_device_map["lm_head"] = available_gpu - 1

    for i in range(available_gpu):
        start = i * num_layers_per_gpu
        end = (i + 1) * num_layers_per_gpu
        if i == available_gpu - 1:
            end = num_layers
        # assign to GPU i
        for j in range(start, end):
            custom_device_map[f"model.layers.{j}"] = i

    return custom_device_map


def load_updated_attention(model):
    layer_idx = 21
    loaded_weight = torch.load(f"attention_modi/att_weights/test/{layer_idx}.pt")['k_proj.weight']

    weight_to_change = model.model.layers[layer_idx].self_attn.k_proj.weight
    weight_dtype = weight_to_change.dtype
    weight_device = weight_to_change.device

    loaded_weight = loaded_weight.to(weight_dtype).to(weight_device)

    weight_replace = nn.Parameter(loaded_weight)
    model.model.layers[layer_idx].self_attn.k_proj.weight = weight_replace


# hook configs: 
attention_layer_out = {}
mlp_layer_out = {}


def post_process_model(llama_model, register_hooks):
    llama_legacy = llama_model.model
    # rewrite forward method using model parallelism, and replace current model's forward method with this one.

    # add hooks to each transformer block; 

    # modify the forward method of decoder layer to implement skip connection; 
    # also need to know which layers to modify the method. 



    # if register_hooks:
    #     for idx, decoder_layer in enumerate(llama_legacy.layers):
    #         attention_layer_out[idx] = []
    #         mlp_layer_out[idx] = []
    #         def get_attention_hook(index):
    #             def attention_hook(module, input, output):
    #                 attention_layer_out[index].append(output[0][:, -1].detach().cpu())  # shape [batch, hidden]
    #             return attention_hook
    #         decoder_layer.self_attn.register_forward_hook(get_attention_hook(idx))
    #         def get_mlp_hook(index):
    #             def mlp_hook(module, input, output):
    #                 mlp_layer_out[index].append(output[:, -1].detach().cpu())  # shape [batch, hidden]
    #             return mlp_hook
    #         decoder_layer.mlp.register_forward_hook(get_mlp_hook(idx))
    

    def model_parallel_forward_causalLM(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):  # this method will mostly be the same as the original forward method in hugging face's 
        # transformers package.
        last_gpu = torch.cuda.device_count() - 1
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            input_ids.to(self.model.embed_tokens.weight.device)
            inputs_embeds = self.model.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds,
            past_key_values_length)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.model.layers):
                        
            if output_hidden_states:
                all_hidden_states += (hidden_states.detach().cpu(), )

            past_key_value = past_key_values[
                idx] if past_key_values is not None else None
            # if past_key_value is not None:
            #     past_key_value[0].to(decoder_layer.device)  # moving key to device
            #     past_key_value[1].to(decoder_layer.device)  # moving value to device
            # hidden_states.to(decoder_layer.device)
            # attention_mask.to(decoder_layer.device)
            # position_ids.to(decoder_layer.device)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer.forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            # move all past_key_values to the same device as well
            if use_cache:
                cache_output = layer_outputs[2 if output_attentions else 1]
                keys = cache_output[0].to(f"cuda:{last_gpu}")
                values = cache_output[1].to(f"cuda:{last_gpu}")
                next_decoder_cache += (
                    (keys, values), )

            if output_attentions:
                all_self_attns += (layer_outputs[1].detach().cpu(), )
        hidden_states.to(self.model.norm.weight.device)
        hidden_states = self.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states.detach().cpu(), )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            outputs = tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        else:
            outputs = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        input_ids = input_ids.to(f"cuda:{last_gpu}")
        logits = logits.to(f"cuda:{last_gpu}")  # for handling device issues, as model.generate is called on cpu, not cuda
        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output
        # breakpoint()
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    llama_model._prepare_decoder_attention_mask = llama_legacy._prepare_decoder_attention_mask

    llama_model.forward = types.MethodType(model_parallel_forward_causalLM,
                                          llama_model)
    llama_model.gradient_checkpointing = False

    attention_hook_processing(llama_model, {0: [value_hook], 1: [value_hook]})

    llama_model.training = False
    


def run_model(model_path="llama_vicuna/13B",
                model_hidden_layers=40,
              tokenizer_path="llama_converted_weights/tokenizer",
              datasets=["custom"],
              project_dir="/home/hujingy5/projects/def-six/hujingy5/model",
              batch_size=4,
              curr_batch_idx=0,
              num_samples=200,
              register_hooks=True,
              ):
    last_gpu = torch.cuda.device_count() - 1

    
    print("loading tokenizer")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        tokenizer_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token = tokenizer.bos_token
    
    stop_token_ids = [tokenizer.convert_tokens_to_ids(x) for x in [
        ['</s>'], ['User', ':'], ['system', ':'], ['▁assistant', ':']
    ]]
    stop_token_ids = [torch.LongTensor(x).to(last_gpu) for x in stop_token_ids]

    # class StopOnTokens(StoppingCriteria):
    #     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    #         # realizing this stopping criteria only works when batch_size is 1...
    #         # Also it seems like if the model is not being stopped, more outputs will still be generated along the way
    #         # Which is extremely terrible in this case. 
    #         # The only option left is to record a set of stopping tokens, to record when each prompt has stopped, 
    #         # and to aid in extracting only the necessary outputs out. 
    #         for stop_ids in stop_token_ids:
    #             if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
    #                 return True
    #         return False
    # stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    custom_device_map = acquire_device_dict(model_hidden_layers)
    model = LlamaForCausalLM.from_pretrained(model_path, 
        # load_in_8bit=True,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        # device_map cannot be auto, or will not follow model parallel's assignments!
        # https://huggingface.co/docs/accelerate/usage_guides/big_modeling
        # device_map="auto",)
        device_map=custom_device_map,)
    post_process_model(model, register_hooks)


    # The line below is the main contribution of this project right now #2023-08-15
    load_updated_attention(model)

    # model.config.eos_token_id = 2
    model.config.pad_token_id = model.config.eos_token_id
    # Now call the model to generate data. Realizing ddp distributes data to different GPUs, but model parallelism
    # distributes the model to different GPUs. Therefore ddp might lead to potential error if data is not being distributed
    # to appropriate GPUs. Thus not implemented.
    model.eval()

        # breakpoint()
    seed = np.random.randint(0, 10000)
    for dataset in datasets:
        print(f"loading {dataset} dataset")
        if dataset != "custom":
            prompts = create_dataloader(project_dir, dataset, seed)[curr_batch_idx:curr_batch_idx+num_samples]
        else:
            prompts = [
                "User: What is the answer of 2 + 2? Assistant: ", 
                "User: What is the answer of 2 * 2? Assistant: ", 
                "User: What is the answer of 22 + 33? Assistant: ", 
                "User: What is the answer of 22 * 33? Assistant: ", 
                "User: What is the answer of 221 + 397? Assistant: ", 
                "User: What is the answer of 220 * 400? Assistant: ", 
                "User: What is the answer of 127 + (12 / 4)? Assistant: ", 
                "User: What is the answer of 127 - (12 * 4)? Assistant: ", 
                "User: What is the answer of (127 - 12) * 4? Assistant: ", 
                "User: What is the answer of 122 - 12 ^ 8? Assistant: ", 
                "User: What is the answer of 122 - (12 / 4 * 3 - 2)? Assistant: ", 
                "User: What is the answer of 10 * (7 * (5 + 3))? Assistant: ", 
                "User: Simplify the expression: x * (y * (z + a)). Assistant: ", 
                "User: Simplify the expression: a * (b / c) * d. Assistant: ", 
                "User: Simplify the expression: (a * b / c) * d. Assistant: ", 
                "User: Simplify the expression: (a - b + c) ^ 2. Assistant: ", 
                "User: Simplify the expression: x(y(z + a)). Assistant: ", 
                "User: Simplify the expression: (ab/c)d. Assistant: ",
                "User: Simplify the expression: x(y(z + b)). Assistant: ", 
                "User: Simplify the expression: a(b/c)d. Assistant: ", 
            ]
        output_dir = os.path.join(project_dir, "outputs_vicuna", dataset)
        prompt_size = len(prompts)
        batch_start = 0

        while batch_start < prompt_size:
            # breakpoint()
            curr_batch = prompts[batch_start:batch_start+batch_size]
            # curr_batch = ["User: answer this question: what does 1+1 equal to? Assistant: ", 
            #                 "User: Hey, are you conscious? Can you talk to me? Assistant: ",]
            inputs = tokenizer(curr_batch, padding=True)
            # print(inputs)
            # breakpoint()
            input_tensor = torch.tensor(inputs.input_ids)
            # concatenate eos token at beginning of each output, or llama 13B model would have trouble generating outputs. 
            # I don't know why EOS tokens are necessary to be placed at the front of each input prompt. 
            # OK, the problem is being fixed, for both 13B and 7B models. 
            # input_tensor = torch.cat([
            #     torch.ones(input_tensor.shape[0], 1, dtype=input_tensor.dtype) * tokenizer.encode(tokenizer.eos_token)[1], 
            #     input_tensor], dim=1)
            # input_mask = torch.tensor(inputs.attention_mask)
            # input_mask = torch.cat([
            #     torch.zeros(input_mask.shape[0], 1, dtype=input_mask.dtype), 
            #     input_tensor], dim=1)
            with torch.no_grad():
                generate_ids = model.generate(input_tensor.to(f"cuda:{last_gpu}"), max_length=input_tensor.shape[1] + 256,
                # generate_ids = model.generate(input_tensor.to(f"cuda:{last_gpu}"), max_length=input_tensor.shape[1] + 40,
                    return_dict_in_generate=True,
                    output_attentions=True, output_hidden_states=True,
                    # do NOT add this line below!!! Otherwise batched generation is disabled!!! 
                    attention_mask=torch.tensor(inputs.attention_mask, dtype=torch.bool),
                    # stopping_criteria=stopping_criteria,
                    bad_words_ids=[[1], [30166], [29871, 30166]],
                    # temperature=1.0,
                    # top_k=40,
                    # num_beams=4,
                    # num_return_sequences=1,
                    # do_sample=True,
                    # top_p=0.75,
                )
                # breakpoint()
                decode_res = tokenizer.batch_decode(
                    generate_ids.sequences,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
            # now save all hidden states, attention outputs, and generate_ids, along with decode_res
            torch.save(generate_ids.attentions[:1], f"{output_dir}/attention/batch_{batch_start+curr_batch_idx}_self.pt")
            torch.save(generate_ids.attentions[1:], f"{output_dir}/attention/batch_{batch_start+curr_batch_idx}.pt")
            torch.save(generate_ids.hidden_states, f"{output_dir}/hidden/batch_{batch_start+curr_batch_idx}.pt")
            torch.save(generate_ids.sequences, f"{output_dir}/res/batch_{batch_start+curr_batch_idx}.pt")
            if register_hooks:
                # post-process hooks
                hooks = {"att": attention_layer_out, "mlp": mlp_layer_out}
                torch.save(hooks, f"{output_dir}/hook/batch_{batch_start+curr_batch_idx}.pt")
                # reset hook
                for key in attention_layer_out.keys():
                    attention_layer_out[key] = []
                    mlp_layer_out[key] = []
            print(f"batch: {batch_start+curr_batch_idx}:{batch_start+batch_size+curr_batch_idx}")
            batch_start += batch_size



# for debugging only
if __name__ == "__main__":
    print("import complete")
    parser = argparse.ArgumentParser(
                    prog='cot generate',
                    description='generate chain of thought reasoning for questions')
    parser.add_argument('--dataset',
                        type=str, default="gsm8k2")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--hooks", action="store_true", default=False)
    parser.add_argument("--batch_start", type=int, default=0)
    parser.add_argument("--sample_num", type=int, default=200)
    args=parser.parse_args()
    run_model(datasets=[args.dataset], batch_size=args.batch_size, register_hooks=args.hooks, 
              num_samples=args.sample_num, curr_batch_idx=args.batch_start)
    # run_model()
