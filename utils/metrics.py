import numpy as np
import torch
from thop import profile
from sklearn.metrics import top_k_accuracy_score

def accuracy_score(true_labels, logits):
    # Top-1 and Top-5 accuracy score
    top_1_acc, top_5_acc = top_k_accuracy_score(true_labels, logits, k = 1), top_k_accuracy_score(true_labels, logits, k = 5)
    return top_1_acc, top_5_acc

def model_metrics(model, img_size, device):
    input = torch.randn(1, 3, img_size, img_size).to(device)
    macs, params = profile(model, inputs = (input, ))
    return 2*macs, params