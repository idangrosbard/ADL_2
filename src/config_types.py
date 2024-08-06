from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict

from src.types import CONFIG_KEYS
from src.types import LR_SCHEDULER
from src.types import OPTIMIZER
from src.types import SAMPLERS
from src.types import T_SAMPLER
from src.types import TimeStep


class UNetConfig(TypedDict):
    depth: int
    n_channels: int
    p_dropout: float
    init_width: int
    width_expansion_factor: int
    n_convs: int
    kernel_size: int
    resblock: bool


class DDPMConfig(TypedDict):
    unet: UNetConfig
    length: int


class DPM_SOLVER_PPConfig(TypedDict):
    pass


class FashionMNISTConfig(TypedDict):
    dim: int
    T: TimeStep
    t_sampler: T_SAMPLER


class SamplerConfig(TypedDict):
    samplers: List[SAMPLERS]
    num_samples: int
    deterministic_sampling: bool


class TrainingConfig(TypedDict):
    batch_size: int
    epochs: int
    seed: int
    gradient_clip_value: Optional[float]
    optimizer_type: OPTIMIZER
    optimizer_params: Dict[str, Any]
    lr_scheduler: Optional[LR_SCHEDULER]
    lr_scheduler_params: Dict[str, Any]
    early_stopping: bool
    early_stopping_patience: int


class Config(TypedDict):
    ddpm: DDPMConfig
    dpm_solver_pp: DPM_SOLVER_PPConfig
    fashion_mnist: FashionMNISTConfig
    sampler: SamplerConfig
    training: TrainingConfig


def get_sub_config(config: Config, key: CONFIG_KEYS) -> Dict[str, Any]:
    return config[key]  # noqa
