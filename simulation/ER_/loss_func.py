import torch
import torch.nn



def NMSE_loss(pred, true, eps=1e-8):
    mse = (pred-true).pow(2).mean()
    power = true.pow(2).mean()
    return mse / (power + eps)

def MSE_loss(pred, true):
    mse = (pred-true).pow(2).mean()
    return mse


def MSE_loss_per_sample(pred, true):
    return (pred-true).pow(2)