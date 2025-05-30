import torch
import numpy as np
import trimesh
from skimage import measure  # Marching Cubes


def create_grid(resolution):
    """
    Creates a 3D grid of points within the unit cube [-0.5, 0.5]^3.

    Args:
        resolution (int): Number of points along each dimension of the grid.

    Returns:
        torch.Tensor: Grid points, shape (resolution^3, 3).
    """
    x = np.linspace(-0.5, 0.5, resolution)
    xx, yy, zz = np.meshgrid(x, x, x)
    grid = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return torch.from_numpy(grid).float()

def concat_latent_and_grid(latent_code, grid_points):
    """
    Concatenates the latent code with the grid points.

    Args:
        latent_code (torch.Tensor): Latent code, shape (latent_size,).
        grid_points (torch.Tensor): 3D grid points, shape (M, 3).
    Returns:
        torch.Tensor: Concatenated tensor, shape (M, latent_size + 3).
    """
    if latent_code.dim() == 1:
        latent_code = latent_code.unsqueeze(0)  # [1, latent_size]
    latent_expand = latent_code.expand(grid_points.shape[0], -1)  # [M, latent_size]
    return torch.cat([latent_expand, grid_points], dim=1)  # [M, latent_size+3]

def reconstruct_mesh(model, grid_points, resolution, center, scale, batch_size, device, latent_code=None):
    """
    Reconstructs a mesh from the predicted SDF values using Marching Cubes.

    Args:
        model (nn.Module): Trained MLP model.
        grid_points (torch.Tensor): 3D grid of points, shape (resolution^3, 3).
        resolution (int): Number of points along each dimension of the grid.
        batch_size (int): Batch size for SDF prediction.
        device (torch.device): Device to run the model on.

    Returns:
        trimesh.Mesh: Reconstructed mesh.
    """

    model.eval()  # Set model to evaluation mode
    sdf = []
    with torch.no_grad():  # Disable gradient calculation
        for i in range(0, grid_points.shape[0], batch_size):
            batch = grid_points[i:i + batch_size].to(device)
            if latent_code is not None:
                batch = concat_latent_and_grid(latent_code.to(device), batch)
            sdf_batch = model(batch).cpu()  # Predict SDF values
            sdf.append(sdf_batch)
    sdf = torch.cat(sdf, dim=0).numpy()
    sdf = sdf.reshape((resolution, resolution, resolution))  # Reshape to 3D grid

    # Marching Cubes
    print("SDF shape: ", sdf.shape)
    print("SDF min:", sdf.min(), "SDF max:", sdf.max())
    sdf_median = np.median(sdf)
    vertices, faces, _, _ = measure.marching_cubes(sdf, level=0)  # level=0 extracts the zero-level set (surface)
    voxel_size = 1.0 / (resolution - 1)
    vertices = vertices * voxel_size - 0.5
    vertices = vertices * scale + center
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh
