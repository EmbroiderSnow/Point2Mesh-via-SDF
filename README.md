# SDF-Regression with position encoding based on fourier feature

This repository implements an MLP-based model for surface reconstruction from point clouds. The training data consists of spatial sample points and their corresponding SDF (Signed Distance Function) values, with the output being the reconstructed shape.

To capture high-frequency geometric details, positional encoding (based on Fourier features) is introduced, effectively restoring high-frequency signals in the reconstructed surface.

## Environment Dependecies

- Python >= 3.8
- PyTorch = 2.6
- CUDAToolkit = 12.6

## How to run the code?

To train the model, run:

```shell
python ./train.py
```

To eval the model, copy the path of the checkpoint you want to eval, and then run:

```shell
python ./eval.py --checkpoint path/to/the/checkpoint
```

You can check the output .obj in `./eval_expirement/MLP-XXX-XXX/output`

To plot the loss curve, copy the path of the log you want to plot, run:

```shell
python ./plot_log.py --log_file path/to/the/log
```

And then you can find a .png, which is the picture of loss curves.

## Network Architecture Configuration

To customize the model, modify parameters in `/model/config.json` or create your own configuration file.A vanilla MLP complement already exists, namd `mlp_config.json`. 

You may specify the configuration file during training using the `--config` argument:

```shell
python ./train.py --config path/to/the/config/file
```

