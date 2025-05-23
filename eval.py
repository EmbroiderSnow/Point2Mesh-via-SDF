import torch
import numpy as np
from skimage import measure  # Marching Cubes
from models.mlp import MLP  # 导入你的 MLP 模型
from utils.eval_utils import *
import argparse
import os
from pathlib import Path
import datetime
import logging
import sys
from utils.data_loader import SDFDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Test SDF prediction and reconstruct mesh using Marching Cubes")
    parser.add_argument('--model_name', type=str, default='MLP', help='model name')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument("--output_path", type=str, default="./output", help="Path to save the reconstructed mesh (output.obj)")
    parser.add_argument("--grid_resolution", type=int, default=128, help="Resolution of the 3D grid for Marching Cubes")
    parser.add_argument("--test", type=bool, default=False, help="Test mode")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # --- SET GPU DEVICE ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CREATE DIR ---
    experiment_dir = Path('./eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.checkpoint, checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # --- LOG ---
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

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
    model = MLP(**config).to(device)  # Initialize the model
    if args.checkpoint is not None:
        print('Load CheckPoint...')
        logger.info('Load CheckPoint')
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print('Please load Checkpoint to eval...')
        sys.exit(0)
        start_epoch = 0
    
    if args.test:
        print('Test mode activated.')
        logger.info('Test mode activated.')
        # --- DATA LOADING ---
        logger.info('Loading dataset ...')
        DATA_PATH = './data'
        # Considering that the dataset is too small, we use Leave-One-Out Cross-Validation to divide the dataset
        DATASET = SDFDataset(DATA_PATH)
        
        trainDataLoader = DataLoader(DATASET, batch_size=1, shuffle=True, num_workers=4)
        test_on_training_points(model, trainDataLoader, device)

    # --- EVAL ---
    logger.info('Start evaluating...')
    print('Start evaluating...')

    model.eval()

    # Create 3D grid
    logger.info('Creating 3D grid...')
    print('Creating 3D grid...')
    grid_points = create_grid(args.grid_resolution).to(device)

    # Reconstruct mesh
    logger.info('Reconstructing mesh...')
    print('Reconstructing mesh...')
    mesh = reconstruct_mesh(model, grid_points, args.grid_resolution, args.batch_size, device)

    # Save mesh
    os.mkdir(args.output_path) if not os.path.exists(args.output_path) else None
    output_file = os.path.join(args.output_path, "output.obj")
    mesh.export(output_file)
    logger.info(f"Reconstructed mesh saved to {output_file}")
    print(f"Reconstructed mesh saved to {output_file}")