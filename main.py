import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_unnorm_asa_new(rel_asa, seq):
    """
    :param asa_pred: The predicted relative ASA
    :param seq_list: Sequence of the protein
    :return: absolute ASA_PRED
    """
    rnam1_std = "ACDEFGHIKLMNPQRSTVWY-X"

    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
               185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1, 1)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))

    max_seq_len = len(seq[0])
    array_list = []
    for i, single_seq in enumerate(list(seq)):
        rel_asa_current = rel_asa[i, :]
        seq_len_diff = max_seq_len - len(single_seq)
        single_seq = single_seq + ("X" * seq_len_diff)
        asa_max = np.array([dict_rnam1_ASA[i] for i in single_seq]).astype(np.float32)
        abs_asa = np.multiply(rel_asa_current.cpu().detach().numpy(), asa_max)
        array_list.append(abs_asa)

    final_array = np.array(array_list)
    return final_array


def get_angle_degree(preds):

    preds = preds * 2 - 1
    preds_sin = preds[:, :, 0]
    preds_cos = preds[:, :, 1]
    preds_angle_rad = np.arctan2(preds_sin, preds_cos)
    preds_angle = np.degrees(preds_angle_rad)
    return preds_angle

ss_conv_3_8_dict = {'X': 'X', 'C': 'C', 'S': 'C', 'T': 'C', 'H': 'H', 'G': 'H', 'I': 'H', 'E': 'E', 'B': 'E'}
SS3_CLASSES = 'CEH'
SS8_CLASSES = 'CSTHGIEB'

def main_class(data_loader, model1, model2, model3, device):
    ss3_pred_list = []
    ss8_pred_list = []
    ss3_prob_list = []
    ss8_prob_list = []
    names_list = []
    seq_list = []

    model1.eval()
    model2.eval()
    model3.eval()

    for i, data in enumerate(tqdm(data_loader)):
        feats, length, name, seq = data

        feats = feats.to(device, dtype=torch.float)

        pred1 = model1(feats, length)
        pred2 = model2(feats, length)
        pred3 = model3(feats, length)

        pred = (pred1 + pred2 + pred3) / 3

        # pred = pred.view(-1, 11)

        # ss3_pred = pred[:, 0:3]
        # ss8_pred = pred[:, 3:]

        # name = list(name)
        # for i, prot_len in enumerate(list(length)):
        #     prot_len_int = int(prot_len)
        #     ss3_pred_single = ss3_pred[:prot_len_int, :]
        #     ss3_pred_single = ss3_pred_single.cpu().detach().numpy()
        #     ss3_indices = np.argmax(ss3_pred_single, axis=1)
        #     ss3_pred_aa = np.array([SS3_CLASSES[aa] for aa in ss3_indices])[:, None]
        #     ss3_pred_list.append(ss3_pred_aa)
        #     ss3_prob_list.append(ss3_pred_single)

        #     ss8_pred_single = ss8_pred[:prot_len_int, :]
        #     ss8_pred_single = ss8_pred_single.cpu().detach().numpy()
        #     ss8_indices = np.argmax(ss8_pred_single, axis=1)
        #     ss8_pred_aa = np.array([SS8_CLASSES[aa] for aa in ss8_indices])[:, None]
        #     ss8_pred_list.append(ss8_pred_aa)
        #     ss8_prob_list.append(ss8_pred_single)
        #     names_list.append(name[i])
        # for seq in list(seq):
        #     seq_list.append(np.array([i for i in seq])[:, None])

    # return names_list, seq_list, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list
    return pred
# end def


def write_csv(class_out, reg_out, save_dir):
    names, seq, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list = class_out
    psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list = reg_out

    for seq, ss3, ss8, asa, hseu, hsed, cn, psi, phi, theta, tau, ss3_prob, ss8_prob, name in zip(seq, ss3_pred_list,
                                                                                                  ss8_pred_list,
                                                                                                  asa_list, hseu_list,
                                                                                                  hsed_list, cn_list,
                                                                                                  psi_list, phi_list,
                                                                                                  theta_list, tau_list,
                                                                                                  ss3_prob_list,
                                                                                                  ss8_prob_list, names):
        data = np.concatenate((seq, ss3, ss8, asa, hseu, hsed, cn, psi, phi, theta, tau, ss3_prob, ss8_prob), axis=1)

        save_path = os.path.join(save_dir, name + ".csv")
        pd.DataFrame(data).to_csv(save_path,
                                  header=["AA", "SS3", "SS8", "ASA", "HseU", "HseD", "CN", "Psi", "Phi", "Theta",
                                          "Tau", "P3C", "P3E", "P3H", "P8C", "P8S", "P8T", "P8H", "P8G",
                                          "P8I", "P8E", "P8B"])
    return print(f'please find the results saved at {save_dir} with .csv extention')


if __name__ == '__main__':
    print("Please run the run_SPOT-1D-LM.sh instead")
