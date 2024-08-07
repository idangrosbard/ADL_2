import json
from argparse import ArgumentParser
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import torch
import torch.multiprocessing as mp

from scripts.create_slurm_file import run_slurm
from src.consts import C_STEPS
from src.consts import IAddArgs
from src.trainer import Trainer
from src.trainer import set_seed
from src.types import ITrainArgs
from src.types import LR_SCHEDULER
from src.types import MODEL
from src.types import SAMPLERS
from src.types import STREnum
from src.utils.experiment_helper import load_config


def enum_values(enum: Type[STREnum]) -> List[str]:
    return [x.value for x in enum]  # noqa


def parse_args() -> Tuple[ITrainArgs, bool, IAddArgs]:
    parser = ArgumentParser()
    parser.add_argument('--config_name', type=str, default='base')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--with_parallel', action='store_true')
    parser.add_argument('--with_slurm', action='store_true')

    parser.add_argument('--T', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--init_width', type=int)
    parser.add_argument('--width_expansion_factor', type=int)
    parser.add_argument('--n_convs', type=int)
    parser.add_argument('--no_resblock', dest='resblock', action='store_false')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_lr', type=float)
    parser.add_argument('--lr_scheduler', type=str, choices=enum_values(LR_SCHEDULER))
    parser.add_argument('--n_step_lr', type=int)
    parser.add_argument('--sampling_freq', type=int)
    parser.add_argument('--model', type=str, choices=enum_values(MODEL), default=MODEL.DDPM.value)
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--model_depth', type=int)
    parser.add_argument('--sampler', type=str, nargs='+', choices=enum_values(SAMPLERS))
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--deterministic_sampling', action='store_true')
    parser.add_argument('--is_ref', action='store_true')

    args = parser.parse_args()

    config = load_config(args.config_name)

    if args.sampling_freq is not None:
        C_STEPS.SAMPLE_STEP = args.sampling_freq

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
    if args.resblock is not None:
        config['ddpm']['unet']['resblock'] = args.resblock
    if args.model_depth is not None:
        config['ddpm']['unet']['depth'] = args.model_depth

    assert args.model is not None, 'Model must be specified'
    for value in enum_values(MODEL):
        if value != args.model and value in config:
            del config[value]  # noqa

    if args.lr is not None:
        config['training']['optimizer_params']['lr'] = args.lr
    if args.max_lr is not None:
        config['training']['lr_scheduler_params']['max_lr'] = args.max_lr
    if args.lr_scheduler is not None:
        config['training']['lr_scheduler'] = args.lr_scheduler
    if args.n_step_lr is not None:
        config['training']['lr_scheduler_params']['step_size'] = args.n_step_lr

    if args.sampler is not None:
        config['sampler']['samplers'] = args.sampler
    if args.n_samples is not None:
        config['sampler']['num_samples'] = args.n_samples
    if args.deterministic_sampling is not None:
        config['sampler']['deterministic_sampling'] = args.deterministic_sampling
    if args.is_ref is not None:
        config['training']['is_ref'] = args.is_ref

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
        train_args: ITrainArgs,
):
    Trainer(
        config_name=train_args.config_name,
        config=train_args.config,
        run_id=train_args.run_id,
    ).train(rank, world_size)


def main_with_slurm(train_args: ITrainArgs, add_args: IAddArgs):
    # if slurm, we do not allow config changes
    assert (
            json.dumps(train_args.config) == json.dumps(load_config(train_args.config_name))
    ), 'Config changes are not allowed with slurm'

    run_slurm(
        train_args,
        add_args
    )


def main_local(train_args: ITrainArgs, add_args: IAddArgs):
    if add_args.with_parallel:
        set_seed(42)  # TODO: Need here? (we do it again after the spawn)
        world_size = torch.cuda.device_count()
        mp.spawn(
            fn=train_one,
            args=(world_size, train_args),
            nprocs=world_size,
        )
    else:
        train_one(
            None,
            None,
            train_args
        )


def main(main_args: ITrainArgs, with_slurm: bool, add_args: IAddArgs) -> None:
    func = main_with_slurm if with_slurm else main_local
    func(train_args=main_args, add_args=add_args)


if __name__ == '__main__':
    main(*parse_args())
