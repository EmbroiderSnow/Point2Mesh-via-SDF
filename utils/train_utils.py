import torch
import os
from pathlib import Path

def sdf_loss(sdf_predicted, sdf_gt, grad_predicted, grad_gt, lambda_param):
    sdf_loss = torch.mean((sdf_predicted - sdf_gt) ** 2)
    grad_loss = torch.mean((grad_predicted - grad_gt) ** 2)
    return sdf_loss + lambda_param * grad_loss

def divide_dataset(dataset):
    """
    Use Leave-One-Out Cross-Validation to divide the dataset
    """
    num_samples = len(dataset)
    indices = list(range(num_samples))
    train_indices = []
    test_indices = []
    
    for i in range(num_samples):
        train_indices.append(indices[:i] + indices[i+1:])
        test_indices.append([indices[i]])
    
    return train_indices, test_indices

def compute_sdf_gradient(sdf_points, sdf_values):
    """
    计算预测的 SDF 值的梯度.

    Args:
        points (torch.Tensor): 查询点的坐标, 形状为 (batch_size, num_points, 3).
        sdf_values (torch.Tensor): 预测的 SDF 值, 形状为 (batch_size, num_points, 1).

    Returns:
        torch.Tensor: 预测的 SDF 梯度, 形状为 (batch_size, num_points, 3).
    """

    # 创建 ones 张量，用于计算梯度
    ones_like_sdf = torch.ones_like(sdf_values)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=sdf_values,
        inputs=sdf_points,
        grad_outputs=ones_like_sdf,  # 相当于 loss 的权重
        create_graph=True,  # 保留计算图，以便后续计算更高阶的梯度
        retain_graph=True,  # 保留计算图，因为 points 可能被多次使用
    )[0]

    return gradients

def save_checkpoint(epoch, loss, model, optimizer, path, modelnet='checkpoint'):
    savepath = Path(path) / f"{modelnet}-{loss:.4f}-{epoch:04d}.pth"
    state = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)
