import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import os

def parse_training_log(log_file_path):
    """
    解析训练日志文件，提取每一步的SDF损失、梯度损失、潜在损失，
    并提取每个 epoch 的平均总损失。
    """
    sdf_losses_per_step = []
    grad_losses_per_step = []
    latent_losses_per_step = []
    
    total_losses_per_epoch = []
    epochs = []

    # 正则表达式用于匹配日志行
    # 匹配 sdf_loss, grad_loss, latent_loss 的行
    step_loss_pattern = re.compile(r"sdf_loss: ([\d.]+), grad_loss: ([\d.]+), latent_loss: ([\d.]+)")
    # 匹配每个 epoch 结束时的总损失行
    total_loss_epoch_pattern = re.compile(r"Loss: ([\d.]+)")
    # 匹配 epoch 开始的行
    epoch_start_pattern = re.compile(r"Epoch (\d+) \(\d+/\d+\):")

    current_epoch = None
    
    with open(log_file_path, 'r') as f:
        for line in f:
            epoch_start_match = epoch_start_pattern.search(line)
            if epoch_start_match:
                current_epoch = int(epoch_start_match.group(1))
                # 只有当成功解析到 epoch 编号时，才添加到 epochs 列表中
                # 这是为了确保 epochs 和 total_losses_per_epoch 长度一致
                # 因为 total_losses_per_epoch 是在 epoch 结束时记录的
                if current_epoch not in epochs:
                     epochs.append(current_epoch)

            step_loss_match = step_loss_pattern.search(line)
            if step_loss_match:
                sdf_losses_per_step.append(float(step_loss_match.group(1)))
                grad_losses_per_step.append(float(step_loss_match.group(2)))
                latent_losses_per_step.append(float(step_loss_match.group(3)))

            total_loss_epoch_match = total_loss_epoch_pattern.search(line)
            if total_loss_epoch_match:
                total_losses_per_epoch.append(float(total_loss_epoch_match.group(1)))

    # 由于你的日志中，每个epoch的个体损失（sdf_loss, grad_loss, latent_loss）在每个step都会打印，
    # 而epoch的总损失只在epoch结束时打印一次。
    # 为了绘图，我们将每个epoch的个体损失进行平均。
    # 由于日志片段不完整，这里假设每次 "Loss: %f" 行之前的所有 "sdf_loss:" 行属于当前 epoch。
    # 鉴于你的日志结构，我们直接绘制所有读取到的 sdf_loss, grad_loss, latent_loss，
    # 而 total_loss 则按 epoch 绘制。这将导致曲线长度不一致，但能反映波动。
    # 更好的做法是平均每个epoch的 individual losses。
    # 在这个脚本中，我将为简化起见，直接绘制每一步的个体损失，并绘制每个epoch的总损失。
    # 请注意：如果你希望绘制的是每个epoch的平均SDF/梯度/潜在损失，需要更复杂的逻辑来分组并求平均。
    # 但从你提供的日志片段来看，它在每个epoch结束时只打印了`Loss:`，而不是每一步的`total_loss`。

    # 由于 "sdf_loss" 行和 "Loss:" 行的频率不同，直接绘制它们会导致X轴（步数/epoch）混乱。
    # 我们可以绘制一个"简要"曲线，即每个epoch的平均sdf_loss等。
    # 或者我们绘制 "Total Loss" 的曲线，以及所有 "sdf_loss", "grad_loss", "latent_loss" 的原始点作为散点图或细线图。

    # 考虑到日志中，每个Epoch有多行sdf_loss, grad_loss, latent_loss，而只有一行总Loss，
    # 最好的方法是计算每个Epoch的平均 sdf_loss, grad_loss, latent_loss。

    # 重新解析，以每个epoch为单位聚合数据
    epoch_data = {}
    current_epoch_num = -1
    
    with open(log_file_path, 'r') as f:
        for line in f:
            epoch_start_match = epoch_start_pattern.search(line)
            if epoch_start_match:
                current_epoch_num = int(epoch_start_match.group(1))
                epoch_data[current_epoch_num] = {
                    'sdf_losses': [],
                    'grad_losses': [],
                    'latent_losses': [],
                    'total_loss': None # Will be set once per epoch
                }
            
            if current_epoch_num != -1: # Ensure we are inside an epoch block
                step_loss_match = step_loss_pattern.search(line)
                if step_loss_match:
                    epoch_data[current_epoch_num]['sdf_losses'].append(float(step_loss_match.group(1)))
                    epoch_data[current_epoch_num]['grad_losses'].append(float(step_loss_match.group(2)))
                    epoch_data[current_epoch_num]['latent_losses'].append(float(step_loss_match.group(3)))

                total_loss_epoch_match = total_loss_epoch_pattern.search(line)
                if total_loss_epoch_match:
                    epoch_data[current_epoch_num]['total_loss'] = float(total_loss_epoch_match.group(1))

    # 计算每个 epoch 的平均值
    epochs_for_plot = []
    avg_sdf_losses = []
    avg_grad_losses = []
    avg_latent_losses = []
    final_total_losses = []

    for epoch_num in sorted(epoch_data.keys()):
        data = epoch_data[epoch_num]
        if data['sdf_losses'] and data['total_loss'] is not None: # Ensure data is present for this epoch
            epochs_for_plot.append(epoch_num)
            avg_sdf_losses.append(sum(data['sdf_losses']) / len(data['sdf_losses']))
            avg_grad_losses.append(sum(data['grad_losses']) / len(data['grad_losses']))
            avg_latent_losses.append(sum(data['latent_losses']) / len(data['latent_losses']))
            final_total_losses.append(data['total_loss'])

    return epochs_for_plot, avg_sdf_losses, avg_grad_losses, avg_latent_losses, final_total_losses

def plot_losses(epochs, sdf_losses, grad_losses, latent_losses, total_losses, output_path='training_loss_curves.png'):
    """
    绘制损失曲线。
    """
    plt.figure(figsize=(12, 6))

    # 在绘制之前，将所有损失值转换为 numpy 数组，方便处理
    sdf_losses = np.array(sdf_losses)
    grad_losses = np.array(grad_losses)
    latent_losses = np.array(latent_losses)
    total_losses = np.array(total_losses)

    # 为了避免 log(0) 或 log(负数)，我们需要处理损失值。
    # 损失值通常是非负的。如果出现0，需要添加一个很小的 epsilon。
    # 这里我们添加一个非常小的常数，以避免 log(0)
    epsilon = 1e-10 # 足够小，不会影响实际值

    plt.plot(epochs, sdf_losses + epsilon, label='Average SDF Loss (per Epoch)', alpha=0.8)
    plt.plot(epochs, grad_losses + epsilon, label='Average Gradient Loss (per Epoch)', alpha=0.8)
    plt.plot(epochs, latent_losses + epsilon, label='Average Latent Loss (per Epoch)', alpha=0.8)
    plt.plot(epochs, total_losses + epsilon, label='Total Loss (per Epoch)', linewidth=2, color='black')

    # plt.plot(epochs, sdf_losses, label='Average SDF Loss (per Epoch)')
    # plt.plot(epochs, grad_losses, label='Average Gradient Loss (per Epoch)')
    # plt.plot(epochs, latent_losses, label='Average Latent Loss (per Epoch)')
    # plt.plot(epochs, total_losses, label='Total Loss (per Epoch)', linewidth=2, color='black')

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Curves Over Epochs')
    plt.legend()
    plt.grid(True)

    # --- 调整纵轴尺度 ---
    # 你需要根据你的实际损失值范围来设置这些值
    # 例如，如果你的损失都在0到0.1之间，可以设置 y_min=0, y_max=0.1
    # 如果梯度损失特别小，你可以根据其范围来设置
    # 这里我给出一个示例，你需要根据实际情况调整
    # 可以先不设置，让matplotlib自动缩放，然后根据生成的图来决定具体的y_min, y_max
    # 或者打印出 sdf_losses, grad_losses, latent_losses, total_losses 的最大最小值来确定
    
    # 示例：假设你希望关注0到0.05之间的变化，或者特定范围
    y_min = min(sdf_losses)
    y_max = max(latent_losses)
    plt.ylim(y_min, y_max) 

    # 为了帮助你选择合适的范围，这里可以打印出各损失的最大最小值
    print(f"SDF Loss Min: {min(sdf_losses):.6f}, Max: {max(sdf_losses):.6f}")
    print(f"Grad Loss Min: {min(grad_losses):.6f}, Max: {max(grad_losses):.6f}")
    print(f"Latent Loss Min: {min(latent_losses):.6f}, Max: {max(latent_losses):.6f}")
    print(f"Total Loss Min: {min(total_losses):.6f}, Max: {max(total_losses):.6f}")

    # 如果你发现某个损失（例如梯度损失）特别小，而其他损失较大，
    # 导致梯度损失曲线被“压扁”，你可以考虑绘制两个Y轴（一个主轴，一个副轴），
    # 或者绘制多张图，每张图关注一个损失。
    # 不过，通常直接调整Y轴范围就足够了。

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Loss curves saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss curves from a log file.")
    parser.add_argument('--log_file', type=str, help='Path to the training log file.')
    parser.add_argument('--output', type=str, default='training_loss_curves.png', 
                        help='Output path for the plot image.')
    
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found at '{args.log_file}'")
    else:
        epochs, sdf_l, grad_l, latent_l, total_l = parse_training_log(args.log_file)
        if not epochs:
            print("No loss data found in the log file. Please check the log file format.")
        else:
            plot_losses(epochs, sdf_l, grad_l, latent_l, total_l, args.output)