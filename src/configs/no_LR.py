from src.config_types import Config
from src.config_types import DDPMConfig
from src.config_types import DPM_SOLVER_PPConfig
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
            depth=4,
            n_channels=1,
            p_dropout=0.1,
            init_width=64,
            width_expansion_factor=2,
            n_convs=1,
            kernel_size=2,
            resblock=True,
            stride=1,
            padding=1,
            upsample_scale_factor=2,
            upsample_mode='bilinear',
        ),
        length=500,
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
        num_samples=2,
        deterministic_sampling=True,
        beta_1=1e-4,
        beta_T=0.02,
        fast_dpm_num_steps=20,
    ),
    training=TrainingConfig(
        batch_size=64,
        epochs=10,
        seed=42,
        gradient_clip_value=True,
        optimizer_type=OPTIMIZER.ADAMW,
        optimizer_params={
            'lr': 1e-4,
        },
        lr_scheduler=None,
        lr_scheduler_params={},
        early_stopping=True,
        early_stopping_patience=3,
        is_ref=False,
    ),
)
