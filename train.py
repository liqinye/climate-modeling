import torch
import matplotlib.pyplot as plt
from datapipe import TrainDataset
from unet import *
from elucidated_diffusion import *
from stats import *
from trainer import *

def train():
    unet = Unet(
        dim = 64,
        dim_mults = (4, 8, 16, 16),
        random_fourier_features = True,
        channels = 5,
        learned_sinusoidal_dim = 64,
        condition = True
    ).cuda()

    '''
    Elucidated Diffusion
    '''
    elucidated_diffusion = ElucidatedDiffusion(
        unet,
        image_size = (24, 32),
        num_sample_steps = 1000,
        channels = 5,
        condition = True
    ).cuda()

    data_path = ("/extra/ucibdl0/shared/data/climate/h3/*.nc", "/extra/ucibdl0/shared/data/climate/h0/*.nc")

    trainer = Trainer(
        elucidated_diffusion,
        data_path,
        train_batch_size = 64,
        train_lr = 4e-5,
        train_num_steps = 200000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                       # turn on mixed precision
        num_samples = 5
    )
    trainer.load(78)

    trainer.train()

if __name__ == "__main__":
    train()