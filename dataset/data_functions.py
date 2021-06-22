import numpy as np
import pandas as pd

def read_list(file_name):
    '''
    returns list of proteins from file
    '''
    with open(file_name, 'r') as f:
        text = f.read().splitlines()
    return text


def read_fasta_file(fname):
    """
    reads the sequence from the fasta file
    :param fname: filename (string)
    :return: protein sequence  (string)
    """
    with open(fname, 'r') as f:
        AA = ''.join(f.read().splitlines()[1:])
    return AA


def one_hot(seq):
    RNN_seq = seq
    BASES = 'ARNDCQEGHILKMFPSTWYV'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])
    return feat


def get_fasta_2d(fname):
    seq = read_fasta_file(fname)
    one_hot_feat = one_hot(seq)
    temp = one_hot_feat[None, :, :]
    temp = np.tile(temp, (temp.shape[1], 1, 1))
    feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)
    return feature

def angle_norm(angle):
    rad_angle = np.deg2rad(angle)
    angle_split = (np.concatenate([np.sin(rad_angle), np.cos(rad_angle)], 1) + 1) / 2.
    return angle_split


rnam1_std = "ACDEFGHIKLMNPQRSTVWY-X"
ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
           185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1, 1)
dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))


def read_spot_single(file_name, seq):
    data = pd.read_csv(file_name)
    ss3_prob = np.concatenate((data['P3C'][:, None], data['P3E'][:, None], data['P3H'][:, None]), 1).astype(np.float32)
    ss8_prob = np.concatenate((
        data['P8C'][:, None], data['P8S'][:, None], data['P8T'][:, None], data['P8H'][:, None], data['P8G'][:, None],
        data['P8I'][:, None], data['P8E'][:, None], data['P8B'][:, None]), 1).astype(np.float32)

    ASA_den = np.array([dict_rnam1_ASA[i] for i in seq]).astype(np.float32)[:, None]
    asa = data['ASA'][:, None]
    asa_relative = np.clip(asa / ASA_den, 0, 1)

    hseu = data['HseU'][:, None]
    hsed = data['HseD'][:, None]
    CN = data['CN'][:, None]

    psi = data['Psi'][:, None]
    psi_split = angle_norm(psi)
    phi = data['Phi'][:, None]
    phi_split = angle_norm(phi)
    theta = data['Theta'][:, None]
    theta_split = angle_norm(theta)
    tau = data['Tau'][:, None]
    tau_split = angle_norm(tau)
    spot_single_feat = np.concatenate(
        (ss3_prob, ss8_prob, asa_relative, hseu, hsed, CN, psi_split, phi_split, theta_split, tau_split), 1)
    return spot_single_feat


def get_feats_2d(feats):
    temp = feats[None, :, :]
    temp = np.tile(temp, (temp.shape[1], 1, 1))
    feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)
    return feature

