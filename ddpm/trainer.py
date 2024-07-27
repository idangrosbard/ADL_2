import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import nn, Tensor, LongTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable


class Trainer(object):
    def __init__(self, model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler, diffusion_process: nn.Module, sampler: nn.Module, sampling_frequency: int, device: torch.device, summary_writer: SummaryWriter, image_denormalize: Callable) -> None:
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.diffusion_process = diffusion_process
        self.diffusion_process.to(device)
        self.device = device
        self.summary_writer = summary_writer
        self.total_steps = 0
        self.sampler = sampler
        self.sampler.to(device)
        self.sampling_freq = sampling_frequency
        self.image_denormalize = image_denormalize


    def batch(self, x_0: Tensor, t: LongTensor, train: bool = True) -> float:
        self.optimizer.zero_grad()
        x_0 = x_0.to(self.device)
        t = t.to(self.device)
        x_t, epsilon = self.diffusion_process.sample(x_0, t)
        print(x_t.device, epsilon.device)
        epsilon_hat = self.model(x_t, t)
        loss = ((epsilon - epsilon_hat) ** 2).mean()

        if train:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss.item()
    

    def epoch(self, data_loader: DataLoader, train: bool = True) -> float:
        loss = 0
        pbar = tqdm(data_loader, desc=f'{"train" if train else "eval"} epoch....')
        for x_0, t in pbar:
            print(x_0.shape, t.shape)
            b_loss = self.batch(x_0, t, train)
            pbar.set_description(f'{"train" if train else "eval"} loss: {b_loss:.4f}')
            self.summary_writer.add_scalar(f'{"train" if train else "eval"}/batch_loss', b_loss, self.total_steps)
            if (self.total_steps % self.sampling_freq) == 0:
                with torch.no_grad():
                    self.sampler.eval()
                    sample = self.sampler(1)
                    self.summary_writer.add_image('sampled image', self.image_denormalize(sample)[0], self.total_steps)
                    self.sampler.train()

            self.total_steps += 1
            loss += b_loss / len(data_loader)

        return loss
    

    def train(self, train_dl: DataLoader, val_dl: DataLoader, n_epochs: int, eval_freq: int) -> float:
        for e in range(n_epochs):
            self.model.train()
            train_loss = self.epoch(train_dl)
            self.summary_writer.add_scalar(f'train/epoch_loss', train_loss, e)
            print(f"Train loss: {train_loss:.4f}")
            if e % eval_freq == 0:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.epoch(val_dl, train=False)
                    self.summary_writer.add_scalar(f'eval/epoch_loss', val_loss, e)
                    print(f"Val loss: {val_loss:.4f}")
