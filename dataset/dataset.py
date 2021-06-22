import torch
from torch.utils.data import Dataset
from dataset.data_functions import read_list, read_fasta_file, get_fasta_2d, read_spot_single, get_feats_2d


class ProteinsDataset(Dataset):
    def __init__(self, list):
        self.prot_list = read_list(list)

    def __len__(self):
        return len(self.prot_list)

    def __getitem__(self, idx):
        fasta_path = self.prot_list[idx]
        prot_name = fasta_path.split('/')[-1].split('.')[0]

        seq = read_fasta_file(fasta_path)
        seq_hot = get_fasta_2d(fasta_path)

        spot_sgl_feat = read_spot_single("data_files/out_spot_1d_sgl/" + prot_name + ".csv", seq)
        #spot_sgl_feat = read_spot_single("/home/jaspreet/jaspreet_data/text_data/model_outs/SPOT-1d-Single_all/" + prot_name + ".csv", seq)
        spot_sgl_2d = get_feats_2d(spot_sgl_feat)

        # features
        feature1 = seq_hot
        feature2 = spot_sgl_2d

        return feature1, feature2, prot_name, seq


def protein_collate_fn(data):
    feature1, feature2, prot_name, seq = zip(*data)
    feats_tensor1 = torch.from_numpy(feature1[0])
    feats_tensor2 = torch.from_numpy(feature2[0])

    return feats_tensor1, feats_tensor2, prot_name, seq  ### also return feats_lengths and label_lengths if using packpadd
