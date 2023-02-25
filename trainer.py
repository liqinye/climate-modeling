import torch
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator
from multiprocessing import cpu_count
from torch.optim import Adam
from ema_pytorch import EMA
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import utils

from utils import *
from stats import *
from datapipe import TrainDataset
import matplotlib.pyplot as plt
import numpy as np



class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 64,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 8e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = (diffusion_model.data_h, diffusion_model.data_w)

        # dataset and dataloader
        data_dir, cond_dir = folder
        self.ds = TrainDataset(data_dir, cond_dir)
        self.ds_maxmin, self.cond_ds_maxmin = self.ds.getMaxMin()
        self.ds.normData(MinMaxNorm(self.ds.getDataset(), self.ds_maxmin), MinMaxNorm(self.ds.getDataset('condition'), self.cond_ds_maxmin))

        train_size = len(self.ds) - 500
        valid_size = 500
        self.train_ds, self.valid_ds = random_split(self.ds, [train_size, valid_size])

        dl = DataLoader(self.train_ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # loss list
        self.loss = []

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def plot(self, image, out_path, num_samples, milestone):
        channels = image.size()[1]
        fig = plt.figure(constrained_layout=True, figsize=(18, 14))
        fig.suptitle(f"sample-{milestone}")
        subfigs = fig.subfigures(nrows=channels, ncols=1)
        title = ['CRM_QV', 'CRM_QC', 'CRM_QI', 'CRM_QPC', 'CRM_QPI']
        vmax = {'CRM_QV': 0.015, 'CRM_QC': 0.001, 'CRM_QI': 0.0005, 'CRM_QPC': 0.0002, 'CRM_QPI': 0.0002}
        for channel, subfig in enumerate(subfigs):
            subfig.suptitle(f'{title[channel]}')
            # create 1x3 subplots per subfig
            axs = subfig.subplots(nrows=1, ncols=num_samples)
            for col, ax in enumerate(axs):
                im = ax.imshow(image[col][channel], interpolation='bilinear',cmap=plt.cm.jet,origin='lower',vmin=0, vmax=vmax[title[channel]])
                subfig.colorbar(im, ax=ax)
        fig.savefig(out_path)
        plt.close("all")

        # grid = int(math.sqrt(num_samples))
        # fig, axes = plt.subplots(figsize=(18, 10),nrows=grid, ncols=grid)
        # for i in range(num_samples):
        #     row = i // grid
        #     col = i % grid
        #     im = axes[row][col].imshow(image[i][0].cpu().detach().numpy(),interpolation='bilinear',cmap=plt.cm.jet,origin='lower', vmin=0, vmax=0.0002)
        #     fig.colorbar(im, ax=axes[row][col])
        # fig.savefig(out_path)
        # plt.close("all")

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for i in range(self.gradient_accumulate_every):
                    data, cond = next(self.dl)
                    data, cond = data.to(device), cond.to(device)
                    with self.accelerator.autocast():
                        loss = self.model(data, cond)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        self.loss.append(total_loss)

                    self.accelerator.backward(loss)              
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.6f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()
                    torch.save(torch.tensor(self.loss), f"{self.results_folder}/loss.pt")
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            valid_data, valid_cond = self.valid_ds[:]
                            valid_cond = valid_cond.to(device)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, ds_maxmin=self.ds_maxmin, cond=valid_cond[:n]), batches))
                            if self.step % (self.save_and_sample_every * 10) == 0:
                                sample = self.ema.ema_model.sample(batch_size=valid_cond.size()[0], ds_maxmin=self.ds_maxmin, cond=valid_cond)
                                sampleStats1(sample, milestone)
                                sampleStats2(sample, milestone)
                        all_images = torch.cat(all_images_list, dim = 0)
                        self.plot(all_images.cpu(), str(self.results_folder / f'sample-{milestone}.png'), self.num_samples, milestone)
                        self.save(milestone)
                        
                pbar.update(1)

        accelerator.print('training complete')