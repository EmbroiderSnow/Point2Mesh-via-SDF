import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SDFDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.uids = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.uids)
    
    def __getitem__(self, idx):
        uid = self.uids[idx]
        pointcloud_path = os.path.join(self.data_dir, uid, 'pointcloud.npz')
        sdf_path = os.path.join(self.data_dir, uid, 'sdf.npz')

        pointcloud_data = np.load(pointcloud_path)
        sdf_data = np.load(sdf_path)

        points = torch.from_numpy(pointcloud_data['points']).float()
        normals = torch.from_numpy(pointcloud_data['normals']).float()
        sdf_points = torch.from_numpy(sdf_data['points']).float()
        sdf_values = torch.from_numpy(sdf_data['sdf']).float()
        sdf_grads = torch.from_numpy(sdf_data['grad']).float()

        # 随机采样 1024 个 SDF 点
        num_samples = sdf_points.shape[0]
        sample_num = min(1024, num_samples)
        indices = torch.randperm(num_samples)[:sample_num]
        sdf_points = sdf_points[indices]
        sdf_values = sdf_values[indices]
        sdf_grads = sdf_grads[indices]
        
        return {
            'pointcloud': points,
            'normals': normals,
            'sdf_points': sdf_points,
            'sdf_values': sdf_values,
            'sdf_grads': sdf_grads
        }
    
if __name__ == "__main__":
    print("Testing SDFDataset...")
    data_dir = '../data'
    dataset = SDFDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print("Pointcloud shape:", batch["pointcloud"].shape)
        print("SDF points shape:", batch["sdf_points"].shape)
        print("SDF values shape:", batch["sdf_values"].shape)
        print("SDF gradients shape:", batch["sdf_grads"].shape)
        
