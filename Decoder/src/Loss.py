import torch.nn as nn


def get_loss(loss_name):  # try L2Loss first then L1 and then SmoothL1 if preditcion is constant
    if type(loss_name) is not str or loss_name not in ["L1Loss", "MSELoss", "KLDivLoss", "BCELoss", "BCEWithLogitsLoss", "HingeEmbeddingLoss", "SmoothL1Loss", "CosineEmbeddingLoss"]:
        raise ValueError("Loss name must be of type string  and match one of the following functions [L1Loss, MSELoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, HingeEmbeddingLoss, SmoothL1Loss, CosineEmbeddingLoss]")

    if loss_name == 'L1Loss':
        loss = nn.L1Loss()
    elif loss_name == 'MSELoss':  # L2 loss
        loss = nn.MSELoss()
    elif loss_name == 'KLDivLoss':
        loss = nn.KLDivLoss()
    elif loss_name == 'BCELoss':
        loss = nn.BCELoss()
    elif loss_name == 'BCEWithLogitsLoss':
        loss = nn.BCEWithLogitsLoss()  # combines Sigmoid layer and BCELoss -> remove Sigmoid from output
    elif loss_name == 'HingeEmbeddingLoss':
        loss = nn.HingeEmbeddingLoss()
    elif loss_name == 'SmoothL1Loss':  # Huber loss
        loss = nn.SmoothL1Loss()
    elif loss_name == 'CosineEmbeddingLoss':
        loss = nn.CosineEmbeddingLoss()

    return loss
