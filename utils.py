import numpy as np
import pandas as pd
import scipy

from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch

def plot_track(track_df, coords = 'xy', **kwargs):
    if coords == 'xy':
        plt.scatter(track_df.x, track_df.y, **kwargs)
        plt.scatter(0, 0, c = 'black', marker = 'x')
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        
    elif coords == 'zr':
        plt.scatter(track_df.z, track_df.r, **kwargs)
        plt.scatter(0, 0, c = 'black', marker = 'x')
        plt.xlabel("z (mm)")
        plt.ylabel("r (mm)")
        
    elif coords == "3d":
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(track_df.x, track_df.y, track_df.z, **kwargs)
        ax.scatter(0, 0, 0, c = 'black', marker = 'x')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')


def get_src_seq(track1_df, track2_df):
    
    mod_info = pd.read_csv("./module_info.csv")
    vocab = np.load("./module_embeddings.npy")
    l1 = len(track1_df)
    l2 = len(track2_df)

    src_df = pd.concat([track1_df, track2_df])
    src_df = src_df.sort_values("r")
    src_df.index = np.arange(src_df.shape[0])
    src = vocab[src_df["um_idx"], :]
    
    return src, src_df 
        
def get_trg_seq(track1_df, track2_df):
    
    mod_info = pd.read_csv("./module_info.csv")
    SOS_IDX = mod_info.shape[0] - 2 # hard coded
    SEP_IDX = mod_info.shape[0] - 1 # hard coded

    modules1 = mod_info.loc[track1_df["um_idx"], ["um_idx", "umid", "cx", "cy", "cz"]]
    modules2 = mod_info.loc[track2_df["um_idx"], ["um_idx", "umid", "cx", "cy", "cz"]]

    sos_hit = pd.DataFrame({"um_idx": [SOS_IDX], "umid":["sos"], "cx": [0], "cy": [0], "cz": [0]})
    sep_hit = pd.DataFrame({"um_idx": [SEP_IDX], "umid":["sep"], "cx": [0], "cy": [0], "cz": [0]})

    if track1_df["r"].iloc[0] < track2_df["r"].iloc[0]: # compare "r" for first hit
        trg_df = pd.concat([sos_hit, modules1, sep_hit, modules2, sep_hit])

    else:
        trg_df = pd.concat([sos_hit, modules2, sep_hit, modules1, sep_hit])
    
    trg_df.index = np.arange(trg_df.shape[0])
    
    return trg_df
    
def load_model(device = 'cpu'):
    model = torch.jit.load("./model_script.pt")
    model.eval()
    model.to(device)
    return model

def greedy_decode(src, src_idxs, model, device = 'cpu'):
    
    vocab = np.load("./module_embeddings.npy")
    vocab_size = len(vocab)
    sos_idx = vocab_size - 2 #hard coded
    sep_idx = vocab_size - 1 #hard coded
    
    count_mask = np.zeros(vocab_size)
    unique, inv, counts = np.unique(src_idxs, return_inverse=True, return_counts=True)  
    count_mask[unique] = counts
    
    count_mask[sos_idx] = 0
    count_mask[sep_idx] = 100
    
    src = torch.as_tensor(src[None, :, :], dtype = torch.float32, device=device) # add batch_dim (batch_size = 1)
    
    pred_idxs_list = [sos_idx]
    
    while True:
    
        pred_idxs = np.array(pred_idxs_list)
        pred_vecs = vocab[pred_idxs, :]
        pred_vecs = np.expand_dims(pred_vecs, axis = 0) # add batch_dim (batch_size = 1)
        pred_vecs = torch.as_tensor(pred_vecs, dtype=torch.float32, device = device)
        
        with torch.no_grad():    
            # preds = model(src, pred_vecs, src_mask = None, trg_mask = None, 
            #               src_key_padding_mask = None, trg_key_padding_mask = None)
            preds = model(src, pred_vecs)
        
        preds = torch.nn.functional.softmax(preds, dim = 2)
        preds = preds[:, -1, :].detach().cpu().numpy()
    
        impossible_preds = count_mask == 0
        preds[0, impossible_preds] = 0 
        
        if len(pred_idxs) == 1: # if this is the first prediction
            preds[:, sep_idx] = 0
        if pred_idxs[-1] == sep_idx: # if model just predicted SEP, it can't predict SEP again
            preds[:, sep_idx] = 0 
            
        if np.count_nonzero(preds) == 0: # no possible "next token"
            break
        
        argmax = np.argmax(preds)
        pred_idxs_list.append(argmax)
        count_mask[argmax] -= 1
    
    return np.array(pred_idxs_list)
    
def visualize_results(pred_idx, gt_idx, mode = "xy"):
    
    mod_info = pd.read_csv("./module_info.csv")
    SEP_IDX = mod_info.shape[0] - 1 # hard coded
    
    gt_sep = np.where(gt_idx == SEP_IDX)[0]
    
    gt_x = np.array(mod_info["cx"])[gt_idx]
    gt_y = np.array(mod_info["cy"])[gt_idx] 
    gt_z = np.array(mod_info["cz"])[gt_idx] 
    gt_r = np.sqrt(gt_x**2 + gt_y**2)
    
    start = 1 # skip SOS token
    for i in range(len(gt_sep)):
        end = gt_sep[i]
        if mode == "xy":
            plt.plot(gt_x[start:end], gt_y[start:end])
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
        elif mode == "zr":
            plt.plot(gt_z[start:end], gt_r[start:end])
            plt.xlabel("z (mm)")
            plt.ylabel("r (mm)")
        start = end + 1 # +1 to skip the SEP token
    
        
    pred_sep = np.where(pred_idx == SEP_IDX)[0]
    
    pred_x = np.array(mod_info["cx"])[pred_idx]
    pred_y = np.array(mod_info["cy"])[pred_idx] 
    pred_z = np.array(mod_info["cz"])[pred_idx] 
    pred_r = np.sqrt(pred_x**2 + pred_y**2)
    
    start = 1 # skip SOS token
    colors = mpl.colormaps["tab10"]
    for i in range(len(pred_sep)):
        end = pred_sep[i]
        color = colors(i)
        if mode == "xy":
            plt.scatter(pred_x[start:end], pred_y[start:end], color=color, s = 150)
        elif mode == "zr":
            plt.scatter(pred_z[start:end], pred_r[start:end], color=color, s = 150)
            
        start = end + 1 # +1 to skip the SEP token
    
    plt.scatter(0,0, c= "black")

def clopper_pearson(passed: float, total: float, level: float = 0.68):
    """
    Estimate the confidence interval for a sampled binomial random variable with Clopper-Pearson.
    `passed` = number of successes; `total` = number trials; `level` = the confidence level.
    The function returns a `(low, high)` pair of numbers indicating the lower and upper error bars.
    """
    alpha = (1 - level) / 2
    lo = scipy.stats.beta.ppf(alpha, passed, total - passed + 1) if passed > 0 else 0.0
    hi = (
        scipy.stats.beta.ppf(1 - alpha, passed + 1, total - passed)
        if passed < total
        else 1.0
    )
    average = passed / total
    return (average - lo, hi - average)


def get_ratio(passed: List[int], total: List[int]):
    if len(passed) != len(total):
        raise ValueError(
            "Length of passed and total must be the same"
            f"({len(passed)} != {len(total)})"
        )

    res = np.array([x / y if y != 0 else 0.0 for x, y in zip(passed, total)])
    error = np.array([clopper_pearson(x, y) for x, y in zip(passed, total)]).T
    return res, error

    
    