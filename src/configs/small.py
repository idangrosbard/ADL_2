from src.config_types import Config
from src.config_types import DDPMConfig, DPM_SOLVER_PPConfig
from src.config_types import FashionMNISTConfig
from src.config_types import SamplerConfig
from src.config_types import TrainingConfig
from src.config_types import UNetConfig
from src.types import LR_SCHEDULER
from src.types import OPTIMIZER
from src.types import SAMPLERS
from src.types import T_SAMPLER
from src.types import TimeStep

config = Config(
    ddpm=DDPMConfig(
        unet=UNetConfig(
            depth=5,
            n_channels=1,
            p_dropout=0.1,
            init_width=64,
            width_expansion_factor=2,
            n_convs=1,
            kernel_size=2,
            resblock=False,
        ),
        length=10_000,
    ),
    dpm_solver_pp=DPM_SOLVER_PPConfig(
    ),
    fashion_mnist=FashionMNISTConfig(
        dim=32,
        T=TimeStep(200),
        t_sampler=T_SAMPLER.UNIFORM,
    ),
    sampler=SamplerConfig(
        samplers=[SAMPLERS.STANDARD, SAMPLERS.FAST_DPM, SAMPLERS.DDIM],
        num_samples=10,
        deterministic_sampling=True,
    ),
    training=TrainingConfig(
        batch_size=64,
        learning_rate=1e-5,
        max_lr=1e-3,
        epochs=10,
        seed=42,
        gradient_clip_value=True,
        optimizer_type=OPTIMIZER.ADAMW,
        optimizer_params={
        },
        lr_scheduler=LR_SCHEDULER.ONE_CYCLE_LR,
        lr_scheduler_params={
            'max_lr': 0.1,
        },
        early_stopping=False,
        early_stopping_patience=5,
    ),
)
