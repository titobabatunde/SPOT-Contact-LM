import time
import torch
import argparse
from torch.utils.data import DataLoader
import tqdm
import esm
# TODO: check text_collate_fun
from dataset.dataset import ProteinsDataset, protein_collate_fn

# BILSTM + MS-RESNET + MS-RES-LSTM = SPOT-1D-SINGLE Model
from models.bilistm import Network
from models.ms_resnet import Network as Network2
from models.ms_res_lstm import Network as Network3
dev = 'cuda:1'

def spot_1d_single(data_loader, model1, model2, model3, device):
    # calls mean ensemble of SPOT-1D-Single
    # ss3_pred_list = []
    # ss8_pred_list = []
    # ss3_prob_list = []
    # ss8_prob_list = []
    # names_list = []
    # seq_list = []

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

model1 = Network()
model2 = Network2()
model3 = Network3()

# load model hyperparameters to cpu
model1.load_state_dict(torch.load("checkpoints/model1.pt", map_location=torch.device('cpu')))
model2.load_state_dict(torch.load("checkpoints/model2.pt", map_location=torch.device('cpu')))
model3.load_state_dict(torch.load("checkpoints/model3.pt",map_location=torch.device('cpu')))

# move model to GPU
model1 = model1.to(dev)
model2 = model2.to(dev)
model3 = model3.to(dev)

# creation for pre-trained model
pre_trained_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
atten_infer = pre_trained_model.to(dev)


