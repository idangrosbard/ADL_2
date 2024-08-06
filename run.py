import json
from argparse import ArgumentParser
from typing import Optional
from typing import Tuple

import torch
import torch.multiprocessing as mp

from scripts.create_slurm_file import run_slurm
from src.config_types import Config
from src.consts import C_STEPS
from src.consts import IAddArgs
from src.trainer import Trainer
from src.trainer import set_seed
from src.types import IConfigName
from src.types import ITrainArgs
from src.types import LR_SCHEDULER
from src.types import MODEL
from src.types import SAMPLERS
from src.utils.experiment_helper import load_config


def parse_args() -> Tuple[ITrainArgs, bool, IAddArgs]:
    parser = ArgumentParser()
    parser.add_argument('--config_name', type=str, default='small')  # TODO: remove default and make it required
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--with_parallel', type=bool, default=False)
    parser.add_argument('--with_slurm', type=bool, default=False)

    parser.add_argument('--T', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--init_width', type=int)
    parser.add_argument('--width_expansion_factor', type=int)
    parser.add_argument('--n_convs', type=int)
    parser.add_argument('--no_resblock', action='store_false')  # TODO: confusing, change to resblocks
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_lr', type=float)
    parser.add_argument('--step_lr_schedule', action='store_true')
    parser.add_argument('--n_step_lr', type=int)
    # TODO: remove from config, compute automatically
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sampling_freq', type=int)
    parser.add_argument('--model', type=str, choices=[x.value for x in MODEL])
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--model_depth', type=int)
    parser.add_argument('--sampler', type=str, nargs='+', choices=[x.value for x in SAMPLERS])
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--deterministic_sampling', action='store_true')

    args = parser.parse_args()

    config = load_config(args.config_name)

    if args.sampling_freq is not None:
        C_STEPS.SAMPLE_STEP = args.sampling_freq

    # TODO: that's for backward compatibility for our code, once you adjust your code to work with config, let's remove this
    if args.T is not None:
        config['fashion_mnist']['T'] = args.T
    if args.input_dim is not None:
        config['fashion_mnist']['dim'] = args.input_dim

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    if args.init_width is not None:
        config['ddpm']['unet']['init_width'] = args.init_width
    if args.width_expansion_factor is not None:
        config['ddpm']['unet']['width_expansion_factor'] = args.width_expansion_factor
    if args.n_convs is not None:
        config['ddpm']['unet']['n_convs'] = args.n_convs
    if args.no_resblock is not None:
        config['ddpm']['unet']['resblock'] = args.no_resblock
    if args.model_depth is not None:
        config['ddpm']['unet']['depth'] = args.model_depth

    # TODO: find better way to handle this
    assert args.model is not None, 'Model must be specified'
    for key in MODEL:
        if key.value != args.model:
            del config[key.value]

    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.max_lr is not None:
        config['training']['max_lr'] = args.max_lr
    if args.step_lr_schedule is not None:
        config['training']['lr_scheduler'] = LR_SCHEDULER.STEP if args.step_lr_schedule else None
    if args.n_step_lr is not None:
        config['training']['lr_scheduler_params']['step_size'] = args.n_step_lr

    if args.sampler is not None:
        config['sampler']['samplers'] = args.sampler
    if args.n_samples is not None:
        config['sampler']['num_samples'] = args.n_samples
    if args.deterministic_sampling is not None:
        config['sampler']['deterministic_sampling'] = args.deterministic_sampling

    return (
        ITrainArgs(
            config_name=args.config_name,
            config=config,
            run_id=args.run_id,
        ),
        args.with_slurm,
        IAddArgs(
            with_parallel=args.with_parallel,
        )
    )


def train_one(
        rank: Optional[int],
        world_size: Optional[int],
        config_name: IConfigName,
        config: Config,
        run_id: Optional[str]
):
    Trainer(
        config_name=config_name,
        config=config,
        run_id=run_id,
    ).train(rank, world_size)


def main_with_slurm(main_args: ITrainArgs, add_args: IAddArgs):
    # if slurm, we do not allow config changes
    assert (
            json.dumps(main_args.config) == json.dumps(load_config(main_args.config_name))
    ), 'Config changes are not allowed with slurm'

    run_slurm(
        main_args,
        add_args
    )


def main_local(main_args: ITrainArgs, add_args: IAddArgs):
    if add_args.with_parallel:
        set_seed(42)  # TODO: Need here? (we do it again after the spawn)
        world_size = torch.cuda.device_count()
        mp.spawn(
            fn=train_one,
            args=(world_size, *main_args),
            nprocs=world_size,
        )
    else:
        train_one(
            None,
            None,
            *main_args
        )


def main(main_args, with_slurm, add_args):
    func = main_with_slurm if with_slurm else main_local
    func(main_args=main_args, add_args=add_args)


if __name__ == '__main__':
    main(*parse_args())
