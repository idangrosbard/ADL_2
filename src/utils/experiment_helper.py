import importlib
import time
from typing import Optional
from typing import Type

import humanize
from typing_extensions import assert_never

from src.consts import FORMATS
from src.types import CONFIG_KEYS
from src.types import IConfigName
from src.config_types import Config
from src.types import MODEL


def load_config(config_name: IConfigName) -> Config:
    # Dynamically import the config
    config_module = importlib.import_module(f'src.configs.{config_name}')
    return config_module.config


def get_config_key_by_arch(arch_name: MODEL) -> CONFIG_KEYS:
    if arch_name == MODEL.DDPM:
        return CONFIG_KEYS.DDPM
    elif arch_name == MODEL.EDM:
        return CONFIG_KEYS.EDM
    assert_never(arch_name)


def construct_experiment_name(
        config_name: IConfigName,
        model_name: MODEL,
) -> str:
    return '__'.join([
        config_name,
        model_name,
    ])


def create_run_id(run_id: Optional[str]) -> str:
    return run_id if (run_id is not None) else time.strftime(FORMATS.TIME)


def get_model_name_from_config(config: Config) -> MODEL:
    config_models = [
        model
        for model in MODEL
        if model.value in config  # noqa
    ]
    assert len(config_models) == 1, f"Found multiple models in config: {config_models}"
    return config_models[0]
