import functools
import json
import os
from argparse import ArgumentParser
from typing import Optional
from typing import Tuple

import torch
import torch.multiprocessing as mp

from scripts.create_slurm_file import run_slurm
from src.config_types import Config
from src.configs.inventory import config_inventory
from src.configs.inventory import modify
from src.consts import IAddArgs
from src.trainer import Trainer
from src.trainer import set_seed
from src.types import ITrainArgs
from src.utils.argparse_utils import add_arguments_from_typed_dict
from src.utils.argparse_utils import update_config_from_args


def parse_args() -> Tuple[ITrainArgs, bool, IAddArgs]:
    parser = ArgumentParser()
    BASE_CONFIG = os.getenv('BASE_CONFIG', 'base')
    parser.add_argument('--config_name', type=str, default=BASE_CONFIG)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--with_parallel', action='store_true')
    parser.add_argument('--with_slurm', action='store_true')

    base_config = config_inventory[BASE_CONFIG]
    add_arguments_from_typed_dict(parser, "", Config, base_config, print_only=False)  # noqa
    args = parser.parse_args()

    config = modify(config_inventory[args.config_name], functools.partial(update_config_from_args, args=args))

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
            json.dumps(train_args.config) == json.dumps(config_inventory[train_args.config_name])
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
    # Use the function to print argparse statements for debugging
    # add_arguments_from_typed_dict(None, "", Config, config_inventory['base'], print_only=True)
    main(*parse_args())
