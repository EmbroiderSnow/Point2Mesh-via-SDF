import os
import torch
import time
import datetime
import json
import logging
import argparse
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from models.mlp import MLP
from utils.data_loader import SDFDataset
from utils.train_utils import *
from tqdm import tqdm

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
    with open('models/config.json', 'r') as f:
        config = json.load(f)
    logger.info(f"config: {config}")
    net_config = config['NetConfig']
    hyperparameter = config['HyperParameter']
    lr = hyperparameter['learning_rate']
    model = MLP(**net_config).to(device)
    num_shapes = len(DATASET)
    latent_size = net_config['latent_size']
    latent_codes = torch.nn.Embedding(num_shapes, latent_size).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(latent_codes.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
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
        loss_record = []

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            sdf_points = data['sdf_points']
            sdf_values = data['sdf_values'].view(1, -1, 1)
            sdf_grads = data['sdf_grads']
            shape_id = data['shape_id'].to(device)
            B, N, _ = sdf_points.shape
            
            # --- SAMPLE ---
            sample_num = hyperparameter['sample_num']
            epsilon = hyperparameter['epsilon']
            surface_rate = hyperparameter['surface_rate']
            alpha = hyperparameter['alpha']
            grad_lambda = hyperparameter['grad_lambda']
            latent_lambda = hyperparameter['latent_lambda']

            surface_mask = (sdf_values.abs() < epsilon).squeeze()
            if surface_mask.dim() == 1:
                surface_mask = surface_mask.unsqueeze(0)

            surface_indices = surface_mask.nonzero(as_tuple=True)
            other_mask = ~surface_mask
            other_indices = other_mask.nonzero(as_tuple=True)

            num_surface = min(surface_indices[0].shape[0], sample_num * surface_rate // 100)
            num_other = sample_num - num_surface

            # Sample surface points
            if num_surface > 0:
                perm = torch.randperm(surface_indices[0].shape[0])[:num_surface]
                sampled_surface = (surface_indices[0][perm], surface_indices[1][perm])
            else:
                sampled_surface = (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))

            # Sample space points
            if other_indices[0].shape[0] > 0:
                perm = torch.randperm(other_indices[0].shape[0])[:num_other]
                sampled_other = (other_indices[0][perm], other_indices[1][perm])
            else:
                sampled_other = (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))

            # Concatenate
            batch_idx = torch.cat([sampled_surface[0], sampled_other[0]])
            point_idx = torch.cat([sampled_surface[1], sampled_other[1]])

            sdf_points = sdf_points[batch_idx, point_idx, :].view(1, -1, 3).to(device)
            sdf_values = sdf_values[batch_idx, point_idx, :].view(1, -1, 1).to(device)
            sdf_grads = sdf_grads[batch_idx, point_idx, :].view(1, -1, 3).to(device)

            sdf_points.requires_grad_(True)
            B, N, _ = sdf_points.shape
            
            # Make input: concatenate latent code and points
            latent_vec = latent_codes(shape_id)
            latent_expand = latent_vec.unsqueeze(1).expand(-1, N, -1)
            mlp_input = torch.cat([latent_expand, sdf_points], dim=2)
            mlp_input = mlp_input.view(B * N, -1)
            sdf_points = sdf_points.view(B * N, 3) 
            sdf_values = sdf_values.view(B * N, 1)  
            sdf_grads = sdf_grads.view(B * N, 3) 

            optimizer.zero_grad()
            sdf_predicted = model(mlp_input)

            # Compute loss
            mlp_input.requires_grad_(True)
            grad_predicted = compute_sdf_gradient(mlp_input, sdf_predicted)
            loss_dict = sdf_loss(
                sdf_predicted= sdf_predicted,
                sdf_gt=sdf_values,
                grad_predicted=grad_predicted,
                grad_gt=sdf_grads,
                latent_vec=latent_vec,
                alpha=alpha,  # alpha for sdf loss
                lambda_param=grad_lambda,  # lambda for gradient loss
                latent_lambda=latent_lambda,  # regularization for latent code
                eps = epsilon
            )
            loss = loss_dict['total_loss']
            sdf_loss_val = loss_dict['sdf_loss']
            grad_loss_val = loss_dict['grad_loss']
            latent_reg_val = loss_dict['latent_reg']
            logger.info(f"sdf_loss: {sdf_loss_val.item()}, grad_loss: {grad_loss_val.item()}, latent_loss: {latent_reg_val.item()}")
            loss_record.append(loss.item())

            loss.backward()
            optimizer.step()
            global_step += 1

        mean_loss = sum(loss_record) / len(loss_record)
        scheduler.step(mean_loss)
        
        # --- SAVING MODEL ---
        if (mean_loss < min_loss and epoch > 5) or (epoch % 10 == 0):
            min_loss = mean_loss
            logger.info('Saving model ...')
            save_checkpoint(global_epoch + 1, loss, model, latent_codes, optimizer, checkpoints_dir, 'MLP')
            print('Saving model ...')

        print('\r Loss: %f' % mean_loss)
        logger.info('Loss: %f', mean_loss)
        global_epoch += 1
    
    print("Min loss: ", min_loss)
    logger.info('Min loss: %f', min_loss)
    print('End of training ...')
    logger.info('End of training ...') 

if __name__ == "__main__":
    args = parse_args()
    main(args)
    