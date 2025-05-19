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
    y, z, x = np.meshgrid(x, x, x)
    grid = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return torch.from_numpy(grid).float()


def reconstruct_mesh(model, grid_points, resolution, batch_size, device):
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
            sdf_batch = model(batch).cpu()  # Predict SDF values
            sdf.append(sdf_batch)
    sdf = torch.cat(sdf, dim=0).numpy()
    sdf = sdf.reshape((resolution, resolution, resolution))  # Reshape to 3D grid

    # Marching Cubes
    print("SDF shape: ", sdf.shape)
    print("SDF min:", sdf.min(), "SDF max:", sdf.max())
    sdf_median = np.median(sdf)
    vertices, faces, _, _ = measure.marching_cubes(sdf, level=0)  # level=0 extracts the zero-level set (surface)
    mesh = trimesh.Mesh(vertices=vertices, faces=faces)
    return mesh

