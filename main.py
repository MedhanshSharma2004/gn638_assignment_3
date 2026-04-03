import os
import argparse
import yaml
import time
import torch
import torchvision
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from models.eca_mobilenetv2 import get_eca_mobilenetv2
from models.eca_resnet import *
from models.resnet import *
from models.mobilenetv2 import get_mobilenetv2
from utils.metrics import accuracy_score, model_metrics
from utils.load_data import get_dataset
from utils.plots import plot_quantities
from utils.transforms import train_transform, val_transform

with open("configs/config.yaml") as f:
    data = yaml.safe_load(f)

BATCH_SIZE = data['batch_size']
SEED = data['seed']
LEARNING_RATE = data['lr']
NUM_WORKERS = data['num_workers']
NUM_CLASSES = data['num_classes']
NUM_EPOCHS = data['num_epochs']
MOMENTUM = data['momentum']
TRAIN_DATA_DIR = data['train_dir']
VAL_DATA_DIR = data['val_dir']
WEIGHT_DECAY = data['weight_decay']
PLOT_SAVE_DIR = data['plot_dir']
MODEL_SAVE_DIR = data['model_save_dir']
IMG_SIZE = data['image_size']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def classification(model, model_name):    
    # Load Dataset
    print("Loading Dataset")
    train_dataset = get_dataset(TRAIN_DATA_DIR, train_transform())
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory = (device.type == 'cuda'))
    
    val_dataset = get_dataset(VAL_DATA_DIR, val_transform())
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, pin_memory = (device.type == 'cuda'))

    # Training params
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)
    
    model.to(device)
    train_acc_1_list, train_acc_5_list = [], []
    val_acc_1_list, val_acc_5_list = [], []
    train_loss_list, val_loss_list = [], []
    train_fps_list, val_fps_list = [], []
    best_acc_1, best_acc_5 = 0, 0

    for epoch in range(NUM_EPOCHS):
        # Training 
        model.train()
        train_loss = 0.0
        train_logits_list = []
        labels_logits_list = []

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        num_samples = 0
    
        for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            num_samples += X_batch.shape[0]

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            train_logits_list.append(y_pred.detach().cpu().numpy())
            labels_logits_list.append(y_batch.cpu().numpy())
        
        if device.type == "cuda":
            torch.cuda.synchronize()    
        end = time.time()
        
        train_fps_list.append(num_samples/(end - start))
        
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        train_logits = np.concatenate(train_logits_list)
        train_labels = np.concatenate(labels_logits_list)
        train_top_1_acc, train_top_5_acc = accuracy_score(train_labels, train_logits)
        train_acc_1_list.append(train_top_1_acc)
        train_acc_5_list.append(train_top_5_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_logits_list = []
        val_labels_list = []
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            num_samples = 0

            for X_batch, y_batch in tqdm(val_loader, desc="Validation", leave=False):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                num_samples += X_batch.shape[0]
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                val_logits_list.append(y_pred.detach().cpu().numpy())
                val_labels_list.append(y_batch.cpu().numpy())
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            val_fps_list.append(num_samples/ (end-start))
            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)

        val_logits = np.concatenate(val_logits_list)
        val_labels = np.concatenate(val_labels_list)
        val_top_1_acc, val_top_5_acc = accuracy_score(val_labels, val_logits)

        if val_top_1_acc > best_acc_1:
            best_acc_1 = val_top_1_acc
            os.makedirs(f"{MODEL_SAVE_DIR}/{model_name}", exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/{model_name}/best_model.pth")
        
        if val_top_5_acc > best_acc_5:
            best_acc_5 = val_top_5_acc

        val_acc_1_list.append(val_top_1_acc)
        val_acc_5_list.append(val_top_5_acc)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {train_loss:.4f}, Training Accuracy (Top-1): {train_top_1_acc:.4f}, Training Accuracy (Top-5): {train_top_5_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy (Top-1): {val_top_1_acc:.4f}, Validation Accuracy (Top-5): {val_top_5_acc:.4f}')

    # Plots
    os.makedirs(f"{PLOT_SAVE_DIR}/{model_name}", exist_ok = True)
    train_val_acc_1_plot_path = os.path.join(PLOT_SAVE_DIR, model_name, "train_val_acc_top_1.png")
    train_val_acc_5_plot_path = os.path.join(PLOT_SAVE_DIR, model_name, "train_val_acc_top_5.png")
    train_val_loss_plot_path = os.path.join(PLOT_SAVE_DIR, model_name, "train_val_loss.png")

    plot_quantities(train_val_acc_1_plot_path, 'Training and Validation Accuracy (Top-1)', train_acc_1_list, val_acc_1_list, 'Training Accuracy (Top-1)', 'Validation Accuracy (Top-1)', 'Epochs', 'Accuracy')
    plot_quantities(train_val_acc_5_plot_path, 'Training and Validation Accuracy (Top-5)', train_acc_5_list, val_acc_5_list, 'Training Accuracy (Top-5)', 'Validation Accuracy (Top-5)', 'Epochs', 'Accuracy')
    plot_quantities(train_val_loss_plot_path, 'Training and Validation Loss', train_loss_list, val_loss_list, 'Training Loss', 'Validation Loss', 'Epochs', 'Loss')
    
    print("-------- Model metrics --------")
    flops, num_params = model_metrics(model, IMG_SIZE, device)
    mean_train_fps, mean_val_fps = np.mean(train_fps_list), np.mean(val_fps_list)
    print(f"FLOPs: {flops}, Number of parameters: {num_params}, Training FPS: {mean_train_fps}, Validation FPS: {mean_val_fps}")
    print(f'\nTop-1 accuracy using {model_name}: {best_acc_1}, Top-5 accuracy using {model_name}: {best_acc_5}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True, choices = ['resnet18', 'resnet50', 'mobilenetv2'])
    parser.add_argument('--eca', action = 'store_true')
    args = parser.parse_args()
    if args.model == "resnet18":
        if args.eca:
            model = get_eca_resnet18(NUM_CLASSES)
        else:
            model = get_resnet18(num_classes=NUM_CLASSES)
    elif args.model == "resnet50":
        if args.eca:
            model = get_eca_resnet50(NUM_CLASSES)
        else:
            model = get_resnet50(num_classes=NUM_CLASSES)
    elif args.model == "mobilenetv2":
        if args.eca:
            model = get_eca_mobilenetv2(NUM_CLASSES)
        else:
            model = get_mobilenetv2(num_classes=NUM_CLASSES)
    else:
        raise ValueError("Invalid model choice")

    print(f"\nUsing model: {args.model} | ECA: {args.eca}\n")
    model_name = f"{args.model}_{'eca' if args.eca else 'baseline'}"
    classification(model, model_name)

    