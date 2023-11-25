# visualization utils
import matplotlib.pyplot as plt
import numpy as np
import torch

def acquire_attention_tensor(attention_path, device):
    """
    Returns a NUMPY attention tensor. 
    """
    attention_tuple = torch.load(attention_path, map_location=device)
    # attention tuple: Tuple[Tuple[Tensor]], outer tuple represents sequences, inner tuple represents layers. Tensor
    # is of size [Batch, Num_heads, query_len, key_len]; realizing query len and key len differ for each sequence. 
    # Therefore to convert to tensor, need padding at key_len dimension to ensure length consistency.  
    attention_seq = []
    for i in range(len(attention_tuple)):
        attention_layer = []
        for j in range(len(attention_tuple[i])):
            attention_layer.append(attention_tuple[i][j].cpu().numpy())
        attention_seq.append(np.stack(attention_layer, axis=0).transpose(1, 0, 2, 3, 4))  # [batch, num_layers, num_heads, query_len, key_len]
    attention_tensor = attention_seq[0]  # [batch, num_layers, num_heads, query_len, key_len]
    for i in range(1, len(attention_seq)):
        # to convert this into tensor, need to perform padding on last dimension of attention_tensor by 1. 
        pad_len = attention_seq[i].shape[-1] - attention_tensor.shape[-1]
        attention_tensor = np.pad(attention_tensor, [(0, 0), (0, 0), (0, 0), (0, 0), (0, pad_len)], mode="constant")
        attention_tensor = np.concatenate([attention_tensor, attention_seq[i]], axis=-2)
    return attention_tensor # [batch, num_layers, num_heads, query_len, key_len]


def plot_max_attention_score(attention_tensor, idx):

    B, L, H, Q, K = attention_tensor.shape
    attention_tensor = attention_tensor.reshape(B, L * H, Q, K)

    # sequences = torch.load(f"outputs_vicuna/gsm8k/res/batch_{idx}.pt", map_location=torch.device('cpu'))
    # seq = tokenizer.batch_decode(
    #         sequences,
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=False)
    attention_plot = attention_tensor.copy()
    attention_plot = np.max(attention_plot, axis=1)
    for batch_idx in range(attention_plot.shape[0]):
        # decoded_seq = []
        # generated_seq = []
        
        # generate_start_idx = K - Q
        # for i in range(sequences.size()[1]):
        #     decoded_seq.append([i, tokenizer.decode(sequences[batch_idx, i])])
        #     if (i - generate_start_idx) > 0:
        #         generated_seq.append([(i - generate_start_idx - 1), tokenizer.decode(sequences[batch_idx, i])])
        # torch.save({"seq": seq, "entire_seq": decoded_seq, "generated_seq": generated_seq}, f"visualizations/gsm8k/seq/{idx}_{batch_idx}.pt")

        # plot the attention matrix for each batch
        plt.imshow(attention_plot[batch_idx])
        plt.colorbar()
        plt.savefig(f"visualizations/gsm8k/img/{idx}_{batch_idx}.png", dpi=254)
        """
        batch_att = attention_tensor[0, :, 40:47]
        batch_att = batch_att.reshape(-1, K)
        max_batch_att = torch.max(batch_att, dim=0)[0]
        torch.argsort(max_batch_att)
        """
        
        # breakpoint()
        # idx_sorted, score_sorted = quantitative_analysis_visualization(attention_plot[batch_idx], start, end)
        print(batch_idx)
        # print(seq[batch_idx])
        # print()
        # print(decoded_seq)
        # print()
        # print(generated_seq)
        # print("_____________________\n\n")
        plt.clf()


def plot_max_attention_score_layer_wise(attention_tensor):
    """
    # input is not torch tensor, but numpy array
    This method, in the future, might be useful for layerwise analysis. 
    """

    # attention_plot = attention_tensor.cpu().numpy()
    attention_plot = attention_tensor.copy()
    for batch_idx in range(attention_plot.shape[0]):
        # plot the attention matrix for each batch
        for layer_idx in range(attention_plot.shape[1]):
            # acquire max attention score for each layer
            attention_plot_layer = np.max(attention_plot[batch_idx, layer_idx], axis=0)
            plt.imshow(attention_plot_layer)
            # title
            plt.title(f"Batch{batch_idx}Layer{layer_idx}")
            plt.colorbar()
            plt.show()

