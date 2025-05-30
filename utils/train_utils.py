import torch
import os
from pathlib import Path

def sdf_loss(sdf_predicted, sdf_gt, grad_predicted, grad_gt, latent_vec, alpha, lambda_param, latent_lambda, eps):
    surface_mask = (sdf_gt.abs() < eps).squeeze()
    weights = torch.exp(-alpha * sdf_gt.abs())
    sdf_loss = torch.mean(weights * (sdf_predicted - sdf_gt) ** 2)
    grad_loss = torch.mean((grad_predicted[surface_mask] - grad_gt[surface_mask]) ** 2) 
    latent_reg = torch.mean(latent_vec ** 2)
    return {
        'sdf_loss': sdf_loss,
        'grad_loss': grad_loss,
        'latent_reg': latent_reg,
        'total_loss': sdf_loss + lambda_param * grad_loss + latent_lambda * latent_reg
    }

def compute_sdf_gradient(mlp_input, sdf_values):
    # 创建 ones 张量，用于计算梯度
    ones_like_sdf = torch.ones_like(sdf_values)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=sdf_values,
        inputs=mlp_input,
        grad_outputs=ones_like_sdf,  # 相当于 loss 的权重
        create_graph=True,  # 保留计算图，以便后续计算更高阶的梯度
        retain_graph=True,  # 保留计算图，因为 points 可能被多次使用
    )[0][:, -3:]

    return gradients

def save_checkpoint(epoch, loss, model, latent_codes, optimizer, path, modelnet='checkpoint'):
    savepath = Path(path) / f"{modelnet}-{loss:.4f}-{epoch:04d}.pth"
    state = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'latent_state_dict': latent_codes.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)
