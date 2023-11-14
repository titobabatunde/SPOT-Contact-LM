import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from dataset.data_functions import read_list, read_fasta_file, get_fasta_2d, read_spot_single, get_feats_2d


class ProteinsDataset(Dataset):
    """
    First one-hot encoding from the protein sequence 
    concatenated to the language model embeddings generated 
    using ESM-1b and ProtTrans models. 
    The one-hot encoding has a dimension of L × 20, 
    where L is the length of the protein. 

    The embedding from ESM-1b is generated from a model 
    trained on the Uniref50 dataset and has a dimension of 
    L × 1280. 
    
    The ProtTrans model was also trained on the Uniref50 
    and employed the T5-XL model to generate an embedding 
    of dimension of L × 1024. 
    [Instead spot_1d_single goes here]
    
    Concatenating all these features yielded the final 
    input features of dimension L × 2324.
    """
    def __init__(self, list):
        self.protein_list = read_list(list)

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        prot_path = self.protein_list[idx]
        # protein name
        protein = prot_path.split('/')[-1].split('.')[0]

        seq = read_fasta_file(prot_path)
        seq_hot = get_fasta_2d(prot_path) # feature 1

        spot_sgl_feat = read_spot_single("data_files/out_spot_1d_sgl/" + protein + ".csv", seq)
        embedding1 = np.load(os.path.join("inputs/", protein + "_esm.npy"))
        #spot_sgl_feat = read_spot_single("/home/jaspreet/jaspreet_data/text_data/model_outs/SPOT-1d-Single_all/" + prot_name + ".csv", seq)
        # feature 2
        spot_sgl_2d = get_feats_2d(spot_sgl_feat)

        return seq_hot, spot_sgl_2d, protein, seq


def protein_collate_fn(data):
    feature1, feature2, prot_name, seq = zip(*data)
    feats_tensor1 = torch.from_numpy(feature1[0])
    feats_tensor2 = torch.from_numpy(feature2[0])

    return feats_tensor1, feats_tensor2, prot_name, seq  ### also return feats_lengths and label_lengths if using packpadd
