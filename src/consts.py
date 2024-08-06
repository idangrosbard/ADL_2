import os
from pathlib import Path
from typing import NamedTuple

from src.types import LR_SCHEDULER
from src.types import SPLIT
from src.types import STEP_TIMING
from src.types import T_SAMPLER


class PATHS:
    PROJECT_DIR = Path(__file__).parent.parent.resolve()
    DATA_DIR = PROJECT_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed'

    CHECKPOINTS_DIR = PROJECT_DIR / 'checkpoints'
    TENSORBOARD_DIR = PROJECT_DIR / 'runs'


class C_DATASETS:
    FASHION_MNIST_DIR = PATHS.PREPROCESSED_DATA_DIR / 'fashionMNIST'
    DEFAULT_T_SAMPLER = T_SAMPLER.UNIFORM
    NUM_WORKERS = 4
    NORMALIZE_MEAN = 0.1307
    NORMALIZE_STD = 0.3081
    IMDB_LRA_SPLIT_NAMES = [SPLIT.TRAIN, SPLIT.TRAIN]


class C_STEPS:
    WARMUP_STEPS = 0
    SAVE_STEP = 1000
    LOG_STEP = 50
    EVAL_STEP = 200
    SAMPLE_STEP = 100
    PRINT_GRAPH = False


class FORMATS:
    TIME = "%Y%m%d_%H-%M-%S"
    LOGGER_FORMAT = '%(asctime)s - %(message)s'


class DDP:
    MASTER_PORT = os.environ.get('MASTER_PORT', '12355')
    MASTER_ADDR = 'localhost'
    BACKEND = 'nccl'
    SHUFFLE = True
    DROP_LAST = True
    NUM_WORKERS = 0


class IAddArgs(NamedTuple):
    with_parallel: bool
    partition: str = 'gpu-a100-killable'
    time: int = 1200
    singal: str = 'USR1@120'
    nodes: int = 1
    ntasks: int = 1
    mem: int = int(5e4)
    cpus_per_task: int = 1
    gpus: int = 2
    account: str = 'gpu-research'
    workspace: Path = PATHS.PROJECT_DIR
    outputs_relative_path: Path = PATHS.TENSORBOARD_DIR.relative_to(PATHS.PROJECT_DIR)
    master_port: str = DDP.MASTER_PORT


STEP_TIMINGS_TO_LR_SCHEDULER = {
    STEP_TIMING.BATCH: [LR_SCHEDULER.ONE_CYCLE_LR],
    STEP_TIMING.EPOCH: [LR_SCHEDULER.STEP],
    STEP_TIMING.EVALUATION: [],
}
