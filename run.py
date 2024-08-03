import diffusion_process
from ddpm import *
from argparse import ArgumentParser, Namespace
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, StepLR
import torch
from torch.utils.tensorboard import SummaryWriter
import ddpm
from denoiser_backbones import get_unet
from datasets import get_dataloaders, denormalize


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--T', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--init_width', type=int, default=64)
    parser.add_argument('--width_expansion_factor', type=int, default=2)
    parser.add_argument('--n_convs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--step_lr_schedule', action='store_true')
    parser.add_argument('--n_step_lr', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sampling_freq', type=int, default=100)
    parser.add_argument('--model', type=str, choices=['edm', 'ddpm'], default='ddpm')
    parser.add_argument('--input_dim', type=int, default=32)
    parser.add_argument('--model_depth', type=int, default=5)
    parser.add_argument('--sampler', type=str, choices=['standard', 'DPMSolver++', 'FastDPM', 'DDIM'], default='standard')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--deterministic_sampling', action='store_true')
    return parser.parse_args()


def main(args: Namespace) -> None:
    # Get dataloaders
    train_dl, val_dl = get_dataloaders(args.batch_size, args.T, args.input_dim)

    # Get model
    if args.model == 'ddpm':
        assert ((torch.log(torch.tensor(args.input_dim)) / torch.log(torch.tensor(2))) // args.model_depth) >= 1, f'Cannot perform more downsampling than input size allows, input_dim={args.input_dim}, model_depth={args.model_depth}'
        
        unet_backbone = get_unet(args.model_depth, 1, 0.1, args.init_width, args.width_expansion_factor, args.n_convs)

        model = ddpm.DDPMModel(unet_backbone, ddpm.PositionalEncoding(args.input_dim, args.T))
        model.to(torch.device(args.device))
        optimizer = AdamW(model.parameters(), lr=args.lr)
        if args.step_lr_schedule:
            scheduler = StepLR(optimizer, args.n_step_lr, gamma=0.1)
        else:
            scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
        summary_writer = SummaryWriter()

        
        betas = diffusion_process.get_betas(args.T)
        sigmas = diffusion_process.get_sigmas(args.T, betas, args.deterministic_sampling)
        alphas = diffusion_process.get_alphas(betas)
        alpha_bar = diffusion_process.get_alphas_bar(alphas)
        dp = diffusion_process.DiffusionProcess(betas, args.input_dim)
        dp.to(torch.device(args.device))

        # Get sampler
        samplers_names = ['standard', 'FastDPM', 'DDIM']
        samplers = []
        for name in samplers_names:
            if name == 'standard':
                sampler = diffusion_process.StandardSampler(model, args.T, betas, args.input_dim, args.deterministic_sampling)
            elif name == 'FastDPM':
                tau = torch.Tensor(list(range(args.T  - 1, 0, -50)[::-1]))
                print(tau)
                tau = tau.long()
                delta_beta = betas[1] - betas[0]
                beta_0 = betas[0]
                sampler = diffusion_process.FastDPM(denoiser=model, shape=args.input_dim, alpha_bar=alpha_bar, delta_beta=delta_beta, beta_0=beta_0, tau = tau)
            elif name == 'DDIM':
                tau = torch.Tensor(list(range(args.T  - 1, 0, -50)[::-1]))
                print(tau)
                tau = tau.long()
                alphas = diffusion_process.get_alphas(betas)
                sampler = diffusion_process.DDIMSampler(model, alphas, tau, etas = torch.ones(args.T - 1), dim=args.input_dim)
            else:
                raise NotImplementedError(f'Sampler {name} not implemented')
            
            sampler.to(torch.device(args.device))
            samplers.append(sampler)
        
        # Get trainer
        trainer = Trainer(model, optimizer, scheduler, dp, samplers, args.sampling_freq, torch.device(args.device), summary_writer, denormalize, args.n_samples)

        # Train
        trainer.train(train_dl, val_dl, args.epochs, args.sampling_freq)
    else:
        raise NotImplementedError(f'Model {args.model} not implemented')


if __name__ == '__main__':
    args = parse_args()
    main(args)