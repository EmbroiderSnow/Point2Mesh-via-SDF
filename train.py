import os
import torch
import time
import datetime
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from models.mlp import MLP
from utils.data_loader import SDFDataset
from utils.train_utils import *
from tqdm import tqdm
import logging
import argparse

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Point2MeshSDF')
    parser.add_argument('--batchsize', type=int, default=1, help='batch size in training')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--model_name', default='Point2MeshSDF', help='model name')
    return parser.parse_args()

def main(args):
    args = parse_args()

    # --- SET DEVICE ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available(), "CUDA is not available!"
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # --- CREATE DIR ---
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_SDF-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # --- LOGGER ---
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    
    # --- DATA LOADING ---
    logger.info('Loading dataset ...')
    DATA_PATH = './data'
    # Considering that the dataset is too small, we use Leave-One-Out Cross-Validation to divide the dataset
    DATASET = SDFDataset(DATA_PATH)
    
    trainDataLoader = DataLoader(DATASET, batch_size=args.batchsize, shuffle=True, num_workers=4)
    
    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- MODEL LOADING ---
    config = {
        "latent_size": 0,
        "dims": [512] * 8,
        "dropout": [2, 4, 6],
        "dropout_prob": 0.2,
        "norm_layers": [1, 2, 3, 4, 5, 6],
        "latent_in": [8],
        "weight_norm": True,
        "xyz_in_all": True,
        "use_tanh": False,
        "latent_dropout": False
    }
    model = MLP(**config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    global_epoch = 0
    global_step = 0
    min_loss = float('inf')
    
    # --- TRAINING ---
    logger.info('Training ...')
    for epoch in range(args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, args.epoch)

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # normals = data['normals'].to(device)
            sdf_points = data['sdf_points'].clone().detach().requires_grad_(True).to(device)
            sdf_values = data['sdf_values'].to(device)
            sdf_grads = data['sdf_grads'].to(device)

            if batch_id == 0 and epoch == 0:
                print("sdf_points min:", sdf_points.min().item(), "max:", sdf_points.max().item())
                print("sdf_values min:", sdf_values.min().item(), "max:", sdf_values.max().item())
                print("sdf_grads min:", sdf_grads.min().item(), "max:", sdf_grads.max().item())
                print("sdf_values mean:", sdf_values.mean().item(), "median:", sdf_values.median().item())

            optimizer.zero_grad()

            if sdf_points.dim() == 3:
                sdf_points = sdf_points.view(-1, 3)
            sdf_predicted = model(sdf_points)
            sdf_values = sdf_values.view(-1, 1)  # 变成 [16384, 1]
            sdf_grads = sdf_grads.view(-1, 3)  # 变成 [16384, 3]
            # print("sdf_predicted shape:", sdf_predicted.shape)
            # print("sdf_values shape:", sdf_values.shape)
            # print("sdf_predicted min:", sdf_predicted.min().item(), "max:", sdf_predicted.max().item())
            grad_predicted = compute_sdf_gradient(sdf_points, sdf_predicted)
            # print("grad_predicted shape:", grad_predicted.shape)
            # print("sdf_grads shape:", sdf_grads.shape)
            # sdf_loss_val = torch.mean((sdf_predicted - sdf_values) ** 2)
            # grad_loss_val = torch.mean((grad_predicted - sdf_grads) ** 2)
            # print(f"sdf_loss: {sdf_loss_val.item()}, grad_loss: {grad_loss_val.item()}")
            loss = sdf_loss(sdf_predicted, sdf_values, grad_predicted, sdf_grads, lambda_param=0.01)

            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} grad mean: {param.grad.abs().mean().item():.6e}, grad max: {param.grad.abs().max().item():.6e}")
            #     else:
            #         print(f"{name} grad is None")
            optimizer.step()
            global_step += 1

        # scheduler.step()
        
        # --- TESTING ---
        if loss < min_loss and epoch > 5:
            min_loss = loss
            logger.info('Saving model ...')
            save_checkpoint(global_epoch + 1, loss, model, optimizer, checkpoints_dir, 'MLP')
            print('Saving model ...')

        print('\r Loss: %f' % loss.item())
        logger.info('Loss: %.2f', loss.item())
        global_epoch += 1
    
    print("Min loss: ", min_loss)
    logger.info('Min loss: %.2f', min_loss)
    print('End of training ...')
    logger.info('End of training ...') 

if __name__ == "__main__":
    args = parse_args()
    main(args)
    