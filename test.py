from datetime import datetime
from torch import nn
import argparse
import os
import time
import torch
import copy
import subprocess
import glob

from models import models
from utils.utils import save_plots
from utils.train import train, validation
from utils.test import test
from dataset import train_dataloader, val_dataloader, test_dataloader


def run_cmd(cmd):
    out = (subprocess.check_output(cmd, shell=True)).decode('utf-8')[:-1]
    return out

def get_free_gpu_indices():
    out = run_cmd('nvidia-smi -q -d Memory | grep -A4 GPU')
    out = (out.split('\n'))[1:]
    out = [l for l in out if '--' not in l]

    total_gpu_num = int(len(out)/5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i*5:i*5+1]][0])

    out = run_cmd('nvidia-smi --query-compute-apps=gpu_bus_id --format=csv')
    gpu_bus_ids_in_use = (out.split('\n'))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]

if len(get_free_gpu_indices()) > 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    ids = ""
    for id in get_free_gpu_indices():
        ids = str(id) + ","
    ids = ids[:-1]
    print(ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ids
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def load_config(config_path="config"):
    """
	This function was copied from Github repository of the BarkNet 1.0 project
	https://github.com/ulaval-damas/tree-bark-classification/blob/master/src/train.py
	"""
    config = open(config_path, 'r')
    config_details = {}
    for line in config:
        if line.find(' = ') != -1:
            name, value = line.split(' = ')
            config_details[name] = value.strip('\n')
    config.close()
    return config_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Please provide a model from the following: MobileNet2, MobileNet3, MobileViT or EfficientNet-B0.')

    parser.add_argument("--model_name")

    args = parser.parse_args()

    model_name = args.model_name

    if model_name not in ['MobileNet2', 'MobileNet3', 'MobileViT', 'EfficientNet-B0']:
        raise ValueError(
            "Please select a model from the following: ['MobileNet2', 'MobileNet3', 'EfficientNet-B0', 'MobileViT'].")

    print("*" * 100)
    print("Testing the model: {}".format(model_name))

    # Get the current date and time to be used in the log and model filenames
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    print("Loading config...")

    # Load configuration
    config_args = load_config()
    num_epochs = int(config_args['N_EPOCHS'])
    batch_size = int(config_args['BATCH_SIZE'])
    learning_rate = float(config_args['LR'])
    weight_decay = float(config_args['WEIGHT_DECAY'])
    lr_decay = float(config_args['LR_DECAY'])
    epoch_decay = int(config_args['EPOCH_DECAY'])
    logs_path = config_args['LOG_PATH']
    dataset_path = config_args['DATASET_PATH']
    beta_1 = float(config_args['BETAS_1'])
    beta_2 = float(config_args['BETAS_2'])
    eps = float(config_args['EPS'])
    amsgrad = bool(config_args['AMSGRAD'])

    path = glob.glob("trained_models/BarkNet_{}*.pth".format(model_name))[0]

    print(path)

    # # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used: {}".format(device))

    print("Loading data using dataloaders...")

    if model_name == "MobileViT":
        input_size = 256
    else:
        input_size = 224

    # # Load data using dataloaders
    test_data = test_dataloader.get_custom_test_dataloader(dataset_path=dataset_path, batch_size=batch_size, input_size= input_size)

    dataset_sizes = {"test": len(test_data.dataset)}

    print("Dataset after split: {}".format(dataset_sizes))

    print("Loading model...")

    model = models.get_model(name=model_name)
    model.to(device)
    model.load_state_dict(torch.load(path))

    top_1_acc, top_5_acc = test(model, test_data, device, model_name)
