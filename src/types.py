from enum import Enum
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import NewType
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypedDict

if TYPE_CHECKING:
    from src.config_types import Config


class STREnum(str, Enum):
    def __str__(self):
        return str(self.value)


class SPLIT(STREnum):
    TRAIN = 'train'
    TEST = 'test'
    EVAL = 'eval'


class T_SAMPLER(STREnum):
    UNIFORM = 'uniform'
    CONSTANT = 'constant'


class SAMPLERS(STREnum):
    STANDARD = 'standard'
    DPM_SOLVER_PP = 'DPMSolver++'
    FAST_DPM = 'FastDPM'
    DDIM = 'DDIM'


class MODEL(STREnum):
    DDPM = 'DDPM'
    EDM = 'EDM'


class CONFIG_KEYS(STREnum):
    SAMPLERS = 'samplers'
    TRAINING = 'training'
    DDPM = MODEL.DDPM.value
    EDM = MODEL.EDM.value
    FASHION_MNIST = 'fashion_mnist'


class STEP_TIMING(STREnum):
    BATCH = 'batch'
    EPOCH = 'epoch'
    EVALUATION = 'evaluation'


class LR_SCHEDULER(STREnum):
    STEP = 'step'
    ONE_CYCLE_LR = 'one_cycle_lr'


class OPTIMIZER(STREnum):
    ADAM = 'adam'
    ADAMW = 'adamw'


class METRICS(STREnum):
    LOSS = 'loss'


IConfigName = NewType('IConfigName', str)
TimeStep = NewType('TimeStep', int)
IEarlyStopped = NewType('IEarlyStopped', bool)
ILoss = NewType('ILoss', float)
IMetrics = NewType('IMetrics', Dict[str, float])


class ITrainArgs(NamedTuple):
    config_name: IConfigName
    config: Config
    run_id: Optional[str]


class Checkpoint(TypedDict):
    epoch: int
    total_steps: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    best_loss: float
