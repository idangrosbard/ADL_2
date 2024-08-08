import copy
from typing import Callable
from typing import Dict
from typing import Optional

from src.config_types import Config
from src.configs.base import base_config
from src.types import LR_SCHEDULER

# IConfigInventoryLeaf = Dict[str, Config]
# IConfigInventory = Dict[str, IConfigInventoryLeaf | Config]
IConfigInventory = Dict[str, Config]
IConfigModifier = Callable[[Config], Config]


def modify(config: Config, func: IConfigModifier) -> Config:
    return func(copy.deepcopy(config))


def modify_lr_scheduler(lr_scheduler: Optional[LR_SCHEDULER], lr_scheduler_params: Dict[str, float]) -> IConfigModifier:
    def modifier(config: Config) -> Config:
        config['training']['lr_scheduler'] = lr_scheduler
        config['training']['lr_scheduler_params'] = lr_scheduler_params
        return config

    return modifier


config_inventory: IConfigInventory = {
    'base': base_config,
    'no_LR_scheduler': modify(base_config, modify_lr_scheduler(None, {})),
    'one_cycle': modify(base_config, modify_lr_scheduler(LR_SCHEDULER.ONE_CYCLE_LR, {'max_lr': 1e-3})),
    'step_LR': base_config,
}
