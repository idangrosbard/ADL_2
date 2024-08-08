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
    EVAL_STEP = 1000
    SAMPLE_STEP = 100
    PRINT_GRAPH = False


class FORMATS:
    TIME = "%Y%m%d_%H-%M-%S"
    LOGGER_FORMAT = '%(asctime)s - %(message)s'


class ENV_VARS:
    BASE_CONFIG = 'BASE_CONFIG'
    MASTER_PORT = 'MASTER_PORT'
    MASTER_ADDR = 'MASTER_ADDR'


class DDP:
    MASTER_PORT = os.environ.get(ENV_VARS.MASTER_PORT, '12355')
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


class TEXTS:
    CHECKPOINT_SAVED = lambda path: f"Checkpoint saved at {path}"
    CHECKPOINT_LOADED = "Checkpoint loaded"
    TRAINING_STARTS = lambda relative_path: f"Training starts, for experiment: {relative_path}"
    TRAINING_ENDS = "Training ends"
    TOTAL_PARAMS = lambda count: f'Total params: {count}'
    EPOCH_INFO = lambda epoch, total_epochs, batch, total_batches, total_steps, loss: ', '.join([
        f'Epoch [{epoch + 1}/{total_epochs}]',
        f'Batch [{batch + 1}/{total_batches}]',
        f'Total Steps: {total_steps}',
        f'Loss: {loss:.4f}'
    ])
    TEST_METRICS = lambda metrics: f'Test Metrics: {metrics}'
    EARLY_STOPPING = "Early stopping triggered"
    EPOCH_TIME = lambda epoch, time: f'Epoch {epoch + 1} took {time:.2f} seconds'
