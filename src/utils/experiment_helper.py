import time
from typing import Optional

from typing_extensions import assert_never

from src.config_types import Config
from src.consts import FORMATS
from src.types import CONFIG_KEYS
from src.types import IConfigName
from src.types import MODEL


def get_config_key_by_arch(arch_name: MODEL) -> CONFIG_KEYS:
    if arch_name == MODEL.DDPM:
        return CONFIG_KEYS.DDPM
    elif arch_name == MODEL.EDM:
        return CONFIG_KEYS.EDM
    assert_never(arch_name)


def construct_experiment_name(
        config_name: IConfigName,
        model_name: MODEL,
        with_ref: bool,
) -> str:
    if with_ref:
        model_name = f'{model_name}_ref'
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
