import os
import esm
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataset import ProteinsDataset, protein_collate_fn


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--save_path', default='results/', type=str, help='save path')
parser.add_argument('--device', default='cpu', type=str,
                    help='"cuda:0", or "cpu" note wont run on other gpu then gpu0 due to limitations of jit trace')
parser.add_argument('--esm_device', default='cpu', type=str,
                    help='"cuda:x", or "cpu" device assignment to ')

args = parser.parse_args()

dataset = ProteinsDataset(args.file_list)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=protein_collate_fn, num_workers=4)
print("dataloader initialised")

### obj creation for pre-trained model
pre_trained_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
atten_infer = pre_trained_model.to(args.esm_device)
print("pretrained model loaded and ")

if args.device == "cpu":
    model_list = ["contact_jits/atten_single_contact_cpu.pth",
                  "contact_jits/atten_only_contact_cpu.pth",
                  "contact_jits/atten_all_contact_cpu.pth",
                  "contact_jits/atten_sgl_dist_cpu.pth",
                  "contact_jits/atten_only_dist_cpu.pth",
                  "contact_jits/atten_all_dist_cpu.pth"]
else:
    model_list = ["contact_jits/atten_single_contact_gpu.pth",
                  "contact_jits/atten_only_contact_gpu.pth",
                  "contact_jits/atten_all_contact_gpu.pth",
                  "contact_jits/atten_sgl_dist_gpu.pth",
                  "contact_jits/atten_only_dist_gpu.pth",
                  "contact_jits/atten_all_dist_gpu.pth"]

for model_num, model_path in enumerate(model_list):

    model = torch.jit.load(model_path)
    # print(f"{model_path} loaded.")
    model = model.to(args.device)
    model.eval()

    for i, data in enumerate(tqdm(dataloader)):
        feat1, feat2, prot_name, seq = data

        feats1 = feat1.to(args.device, dtype=torch.float)[None, :, :, :]
        feats2 = feat2.to(args.device, dtype=torch.float)[None, :, :, :]
        data = [(prot_name[0], seq[0])]

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = pre_trained_model(batch_tokens.to(args.esm_device), repr_layers=[33],
                                        return_contacts=True)
        attention = results["attentions"]

        ### removing eol and sol tokens
        attention = attention[:, :, :, :-1, :-1]  # removing eol token
        attention = attention[:, :, :, 1:, 1:]  # removing sol token

        if model_num == 0 or model_num == 3:
            ## seperating the attention maps of last layer only for model 1 and model 4
            attention = attention[:, -1, :, :, :]
            attention = apc(symmetrize(attention))
            attention = attention.permute(0, 2, 3, 1)
        else:
            ## processing
            batch_size, layers, heads, seqlen, seqlen2 = attention.size()
            attention = attention.view(batch_size, layers * heads, seqlen, seqlen)
            attention = apc(symmetrize(attention))
            attention = attention.permute(0, 2, 3, 1)

        atten_feats = attention.to(args.device, dtype=torch.float)

        if model_num == 2 or model_num == 5:
            all_feats = torch.cat((atten_feats, feats1, feats2), 3)
        else:
            all_feats = atten_feats

        del attention

        if model_num == 0:
            pred = model(all_feats).squeeze(-1)
        elif model_num == 1:
            pred = model(all_feats).squeeze(-1)
        elif model_num == 2:
            pred = model(all_feats).squeeze(-1)
        elif model_num == 3:
            dist_pred1, _, _, _ = model(all_feats)
            dist_pred1 = F.softmax(dist_pred1, dim=3)
            contact_bins_pred1 = dist_pred1[:, :, :, 1:13]
            pred = torch.sum(contact_bins_pred1, axis=3)
            del dist_pred1
        elif model_num == 4:
            dist_pred2, _, _, _ = model(all_feats)
            dist_pred2 = F.softmax(dist_pred2, dim=3)
            contact_bins_pred2 = dist_pred2[:, :, :, 1:13]
            pred = torch.sum(contact_bins_pred2, dim=3)
            del dist_pred2

        elif model_num == 5:
            dist_pred3, _, _, _ = model(all_feats)
            dist_pred3 = F.softmax(dist_pred3, dim=3)
            contact_bins_pred3 = dist_pred3[:, :, :, 1:13]
            pred = torch.sum(contact_bins_pred3, dim=3)
            del dist_pred3

        np.save(args.save_path + prot_name[0] + "_" + str(model_num) + ".npy", pred.cpu().detach().numpy())

    del model

for i, data in enumerate(tqdm(dataloader)):
    feat1, feat2, prot_name, seq = data
    pred1 = np.load(args.save_path + prot_name[0] + "_" + "0" + ".npy")
    pred2 = np.load(args.save_path + prot_name[0] + "_" + "1" + ".npy")
    pred3 = np.load(args.save_path + prot_name[0] + "_" + "2" + ".npy")
    pred4 = np.load(args.save_path + prot_name[0] + "_" + "3" + ".npy")
    pred5 = np.load(args.save_path + prot_name[0] + "_" + "4" + ".npy")
    pred6 = np.load(args.save_path + prot_name[0] + "_" + "5" + ".npy")

    pred_all = np.concatenate((pred1, pred2, pred3, pred4, pred5, pred6), axis=0)
    pred = np.mean(pred_all, axis=0)
    np.save(args.save_path + prot_name[0] + ".npy", pred)
