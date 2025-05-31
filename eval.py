import torch
import json
import argparse
import os
import datetime
import sys

from pathlib import Path
from models.mlp import MLP
from utils.eval_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Test SDF prediction and reconstruct mesh using Marching Cubes")
    parser.add_argument('--model_name', type=str, default='MLP', help='model name')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument("--grid_resolution", type=int, default=128, help="Resolution of the 3D grid for Marching Cubes")
    parser.add_argument("--test", type=bool, default=False, help="Test mode")
    parser.add_argument('--config', type=str, default='model/config.json', help='Configuration file')
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
    output_dir = Path(str(file_dir) + '/output')

    # --- MODEL LOADING ---
    with open(args.config, 'r') as f:
        config = json.load(f)
    net_config = config['NetConfig']
    num_shapes = config['ShapeNum']
    latent_size = net_config['latent_size']
    latent_codes = torch.nn.Embedding(num_shapes, latent_size).to(device)
    model = MLP(**net_config).to(device)  # Initialize the model
    if args.checkpoint is not None:
        print('Load CheckPoint...')
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state_dict"])
        latent_codes.load_state_dict(checkpoint["latent_state_dict"])
    else:
        print('Please load Checkpoint to eval...')
        sys.exit(0)
        start_epoch = 0

    # --- EVAL ---
    print('Start evaluating...')
    parameter_num = sum(p.numel() for p in model.parameters())
    print('Parameter number of the model: ', parameter_num)

    model.eval()

    # Create 3D grid
    print('Creating 3D grid...')
    grid_points = create_grid(args.grid_resolution).to(device)

    for shape_id in range(num_shapes):
        # Reconstruct mesh
        print('Reconstructing mesh...')
        with open('norm_params.json', 'r') as f:
            norm_dict = json.load(f)
        params = norm_dict[str(shape_id)]
        center = np.array(params['center'])
        scale = params['scale']
        latent_code = latent_codes(torch.tensor([shape_id], device=device))  # [1, latent_size]
        mesh = reconstruct_mesh(model, grid_points, args.grid_resolution, center, scale, args.batch_size, device, latent_code=latent_code)

        # Save mesh
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"output-{shape_id}.obj")
        mesh.export(output_file)
        print(f"Reconstructed mesh saved to {output_file}")