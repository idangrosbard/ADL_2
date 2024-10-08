import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import humanize
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import assert_never

from src.config_types import Config
from src.config_types import FashionMNISTConfig
from src.config_types import SamplerConfig
from src.config_types import TrainingConfig
from src.config_types import get_sub_config
from src.consts import C_STEPS
from src.consts import DDP
from src.consts import ENV_VARS
from src.consts import FORMATS
from src.consts import PATHS
from src.consts import STEP_TIMINGS_TO_LR_SCHEDULER
from src.consts import TEXTS
from src.datasets.base_diffusion_dataset import DiffusionDatasetFactory
from src.datasets.fashionMNIST import FashionMNISTDatasetFactory
from src.models.abstract_diffusion_model import AbstractDiffusionModel
from src.models.ddpm import DDPMModel
from src.models.diffusion_process import DiffusionProcess
from src.models.ref_ddpm.ddpm import RefDDPMModel
from src.samplers import dp_utils
from src.samplers.abstract_sampler import AbstractSampler
from src.samplers.ddim import DDIMSampler
from src.samplers.fast_dpm import FastDPM
from src.samplers.standard_sampler import StandardSampler
from src.types import Checkpoint
from src.types import IConfigName
from src.types import IEarlyStopped
from src.types import IMetrics
from src.types import LR_SCHEDULER
from src.types import METRICS
from src.types import MODEL
from src.types import OPTIMIZER
from src.types import SAMPLERS
from src.types import SPLIT
from src.types import STEP_TIMING
from src.utils.experiment_helper import construct_experiment_name
from src.utils.experiment_helper import create_run_id
from src.utils.experiment_helper import get_config_key_by_arch
from src.utils.experiment_helper import get_model_name_from_config
from src.utils.seed import set_seed


def setup(rank: int, world_size: int) -> None:
    os.environ[ENV_VARS.MASTER_ADDR] = DDP.MASTER_ADDR
    os.environ[ENV_VARS.MASTER_PORT] = DDP.MASTER_PORT
    dist.init_process_group(DDP.BACKEND, rank=rank, world_size=world_size)


class Trainer:
    def __init__(
            self,
            config_name: IConfigName,
            config: Config,
            run_id: Optional[str] = None
    ):
        self._run_id = create_run_id(run_id)
        self.config_name = config_name
        self._config = config

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        # add time prefix to log
        formatter = logging.Formatter(FORMATS.LOGGER_FORMAT)
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

        self.best_loss = float('inf')
        self.total_steps = 0
        self.early_stopping_counter = 0

    def configure_logging(self) -> None:
        (PATHS.TENSORBOARD_DIR / self.relative_path).mkdir(parents=True, exist_ok=True)
        if self.is_master_process:
            self.logger.addHandler(logging.StreamHandler())
            self.logger.addHandler(
                logging.FileHandler(PATHS.TENSORBOARD_DIR / self.relative_path / f'{self._run_id}.log'))
            self.dump_config()
        else:
            self.logger.addHandler(logging.NullHandler())

    @property
    def relative_path(self) -> str:
        prefix = construct_experiment_name(
            self.config_name,
            self.model_name,
            self.training_config['with_ref']
        )
        return f"{prefix}/{self._run_id}"

    @property
    def training_config(self) -> TrainingConfig:
        return self._config['training']

    @property
    def samplers_config(self) -> SamplerConfig:
        return self._config['sampler']

    @property
    def dataset_config(self) -> FashionMNISTConfig:
        return self._config['fashion_mnist']

    def dump_config(self) -> None:
        path = PATHS.TENSORBOARD_DIR / self.relative_path / f'config_{time.strftime(FORMATS.TIME)}.json'
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self._config, f, indent=4)

    def save_checkpoint(
            self,
            model: AbstractDiffusionModel,
            optimizer: optim.Optimizer,
            epoch: int,
    ):
        path = PATHS.CHECKPOINTS_DIR / self.relative_path / f'{self.total_steps}.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = Checkpoint(
            epoch=epoch,
            total_steps=self.total_steps,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            best_loss=self.best_loss,
        )
        torch.save(checkpoint, path)

        self.logger.info(TEXTS.CHECKPOINT_SAVED(path))

    def load_checkpoint(self, device: torch.device) -> Optional[Checkpoint]:
        if (checkpoint_path := self.get_latest_chkpt()) is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            return checkpoint
        return None

    def get_latest_chkpt(self) -> Optional[Path]:
        dir_path = PATHS.CHECKPOINTS_DIR / self.relative_path
        if dir_path.exists():
            return max(dir_path.iterdir(), key=lambda x: int(x.stem))
        return None

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def is_master_process(self) -> bool:
        return self._rank == 0

    @property
    def is_distributed(self) -> bool:
        return self._world_size > 1

    def get_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        optimizer_type = self.training_config['optimizer_type']
        optimizer_params = self.training_config['optimizer_params']

        if optimizer_type == OPTIMIZER.ADAM:
            return optim.Adam(
                model.parameters(),
                **optimizer_params
            )
        elif optimizer_type == OPTIMIZER.ADAMW:
            return optim.AdamW(
                model.parameters(),
                **optimizer_params,
            )

        assert_never(optimizer_type)

    def get_lr_scheduler(self, optimizer: optim.Optimizer) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        scheduler_type = self.training_config['lr_scheduler']

        if scheduler_type is None:
            return None
        elif scheduler_type == LR_SCHEDULER.STEP:
            return optim.lr_scheduler.StepLR(optimizer, **self.training_config['lr_scheduler_params'])
        elif scheduler_type == LR_SCHEDULER.ONE_CYCLE_LR:
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                epochs=self.training_config['epochs'],
                steps_per_epoch=len(self._get_dataset_loader(SPLIT.TRAIN)),
                **self.training_config['lr_scheduler_params']
            )

        assert_never(scheduler_type)

    @staticmethod
    def get_loss_fn() -> nn.Module:
        return nn.MSELoss()

    def get_samplers(self, device) -> Tuple[List[AbstractSampler], DiffusionProcess]:
        samplers: List[AbstractSampler] = []

        betas = dp_utils.get_betas(
            self.dataset_config['T'], self.samplers_config['beta_1'], self.samplers_config['beta_T']
        )
        sigmas = dp_utils.get_sigmas(self.dataset_config['T'], betas, self.samplers_config['deterministic_sampling'])
        alphas = dp_utils.get_alphas(betas)
        alpha_bar = dp_utils.get_alphas_bar(alphas)
        dp = DiffusionProcess(betas, self.dataset_config['dim'])
        dp.to(device)

        sampler: AbstractSampler
        for sampler_name in self.samplers_config['samplers']:
            if sampler_name == SAMPLERS.STANDARD:
                sampler = StandardSampler(
                    T=self.dataset_config['T'],
                    betas=betas,
                    shape=self.dataset_config['dim'],
                    deterministic=self.samplers_config['deterministic_sampling']
                )
            elif sampler_name == SAMPLERS.FAST_DPM:
                tau = torch.Tensor(list(range(self.dataset_config['T'] - 1, 0, -50)[::-1]))
                print(tau)
                tau = tau.long()
                delta_beta = betas[1] - betas[0]
                beta_0 = betas[0]

                sampler = FastDPM(
                    shape=self.dataset_config['dim'],
                    alpha_bar=alpha_bar,
                    delta_beta=delta_beta,
                    beta_0=beta_0,
                    tau=tau,
                    eta=None,
                    n_steps=self.samplers_config['fast_dpm_num_steps']
                )
            elif sampler_name == SAMPLERS.DDIM:
                tau = torch.Tensor(list(range(self.dataset_config['T'] - 1, 0, -50)[::-1]))
                print(tau)
                tau = tau.long()
                alphas = dp_utils.get_alphas(betas)

                sampler = DDIMSampler(
                    alphas=alphas,
                    taus=tau,
                    etas=torch.ones(self.dataset_config['T'] - 1),
                    dim=self.dataset_config['dim']
                )
            elif sampler_name == SAMPLERS.DPM_SOLVER_PP:
                raise NotImplementedError()
            else:
                assert_never(sampler_name)
            sampler.to(device)
            samplers.append(sampler)

        return samplers, dp

    @property
    def model_name(self) -> MODEL:
        return get_model_name_from_config(self._config)

    def _get_dataset_loader(self, split: SPLIT) -> DataLoader:
        # cache the dataset
        data_loader_attr = f'_data_loader_{split}'
        if hasattr(self, data_loader_attr):
            return getattr(self, data_loader_attr)

        dataset_wrapper = FashionMNISTDatasetFactory(config=self.dataset_config)
        dataset = dataset_wrapper.get_dataset(split)
        with_shuffle = split == SPLIT.TRAIN

        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=with_shuffle,
                drop_last=DDP.DROP_LAST
            )
            data_loader = DataLoader(
                dataset,
                batch_size=self.training_config['batch_size'] // self.world_size,
                sampler=sampler,
                num_workers=DDP.NUM_WORKERS,
                worker_init_fn=lambda x: set_seed(self.training_config['seed']),
                # collate_fn=dataset.collate_fn
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=with_shuffle
            )

        setattr(self, data_loader_attr, data_loader)
        return data_loader

    def _init_diffusion_model(self) -> AbstractDiffusionModel:
        if self.model_name == MODEL.DDPM:
            model = RefDDPMModel if self.training_config['with_ref'] else DDPMModel
            return model(
                config=get_sub_config(self._config, get_config_key_by_arch(self.model_name)),
                dataset_config=self.dataset_config,
            )
        elif self.model_name == MODEL.EDM:
            return ...
        assert_never(self.model_name)

    def train(self, rank=0, world_size=1) -> Dict[str, Any]:
        set_seed(self.training_config['seed'])
        self._rank = rank or 0  # noqa
        self._world_size = world_size or 1  # noqa

        if self.is_distributed:
            setup(rank, world_size)
            device = torch.device(rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.configure_logging()

        writer = SummaryWriter(
            log_dir=str(PATHS.TENSORBOARD_DIR / self.relative_path)) if self.is_master_process else None
        diffusion_model = self._init_diffusion_model()

        checkpoint = self.load_checkpoint(device)

        dataset_wrapper = FashionMNISTDatasetFactory(config=self.dataset_config)
        samplers, dp = self.get_samplers(device=device)

        optimizer = self.get_optimizer(diffusion_model)
        lr_scheduler = self.get_lr_scheduler(optimizer)

        start_epoch = 0
        self.logger.info(TEXTS.TRAINING_STARTS(self.relative_path))

        if checkpoint is not None:
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move the optimizer state to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            start_epoch = checkpoint['epoch']
            self.total_steps = checkpoint['total_steps']
            self.best_loss = checkpoint['best_loss']
            self.logger.info(TEXTS.CHECKPOINT_LOADED)

        self.logger.info(TEXTS.TOTAL_PARAMS(humanize.intcomma(diffusion_model.count_params())))

        metrics = self._train(
            diffusion_model=diffusion_model,
            dp=dp,
            writer=writer,
            device=device,
            start_epoch=start_epoch,
            dataset_wrapper=dataset_wrapper,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            samplers=samplers,
        )

        return metrics

    def _lr_scheduler_step(
            self,
            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
            step_timing: STEP_TIMING,
            writer: SummaryWriter,
            loss: Optional[float] = None,
    ) -> None:
        scheduler_type = self.training_config['lr_scheduler']
        if scheduler_type is None:
            return

        if scheduler_type in STEP_TIMINGS_TO_LR_SCHEDULER[step_timing]:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                assert loss is not None, "Loss must be provided for ReduceLROnPlateau scheduler"
                scheduler.step(loss)
            else:
                scheduler.step()

            if self.is_master_process:
                writer.add_scalar(
                    'LR',
                    scheduler.get_last_lr()[0],
                    self.total_steps
                )

    @property
    def should_print_to_screen(self) -> bool:
        return self.is_master_process

    def _log_train_metrics(self, loss: float, writer: SummaryWriter, epoch: int, batch: int, total_batches: int):
        if self.is_master_process:
            # Log the training loss
            if self.total_steps % C_STEPS.LOG_STEP == 0:
                writer.add_scalar(
                    'Loss',
                    loss,
                    self.total_steps
                )

                if not self.should_print_to_screen:
                    self.logger.info(TEXTS.EPOCH_INFO(
                        epoch, self.training_config['epochs'], batch, total_batches, self.total_steps, loss
                    ))

    def _log_eval_metrics(self, metrics: IMetrics, writer: SummaryWriter):
        if self.is_master_process:
            for key, value in metrics.items():
                writer.add_scalar(
                    'test_' + key,
                    value,
                    self.total_steps,
                )

                self.logger.info(TEXTS.TEST_METRICS(metrics))

    def _sampling_step(
            self,
            model: AbstractDiffusionModel,
            samplers: List[AbstractSampler],
            n_samples: int,
            dataset_wrapper: DiffusionDatasetFactory,
            writer: SummaryWriter
    ):
        if self.is_master_process and self.total_steps % C_STEPS.SAMPLE_STEP == 0:
            with torch.no_grad():
                for sampler in samplers:
                    sampler.eval()
                    sample = sampler(model, n_samples)
                    for i in range(n_samples):
                        writer.add_image(
                            f'{type(sampler)}_{i}',
                            dataset_wrapper.denormalize(sample[i]),
                            self.total_steps
                        )
                    sampler.train()

    def _save_checkpoint_step(self, model: AbstractDiffusionModel, optimizer: optim.Optimizer, epoch: int) -> None:
        if self.is_master_process:
            if self.total_steps >= C_STEPS.WARMUP_STEPS and self.total_steps % C_STEPS.SAVE_STEP == 0:
                self.save_checkpoint(model, optimizer, epoch)

    def _early_stopping(self, loss: float) -> bool:
        if self.is_master_process:
            self.best_loss = min(self.best_loss, loss)
            if self.training_config['early_stopping']:
                # Early stopping
                early_stopping_patience = self.training_config['early_stopping_patience']
                if loss > self.best_loss:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= early_stopping_patience:
                        self.logger.info(TEXTS.EARLY_STOPPING)
                        return True
                else:
                    self.early_stopping_counter = 0
        return False

    def _aggregate_metrics(self, metrics: IMetrics, device: torch.device) -> IMetrics:
        if self.is_distributed:
            reduced_metrics = torch.tensor(list(metrics.values())).to(device)
            dist.reduce(
                reduced_metrics,
                dst=(int(DDP.MASTER_PORT) % self.world_size),
                op=dist.ReduceOp.SUM
            )
            if self.is_master_process:
                metrics = IMetrics({
                    key: value.item() / self.world_size
                    for key, value in zip(metrics.keys(), reduced_metrics)
                })

        return metrics

    def _train(
            self,
            diffusion_model: AbstractDiffusionModel,
            dp: DiffusionProcess,
            writer: SummaryWriter,
            device: torch.device,
            start_epoch: int,
            dataset_wrapper: DiffusionDatasetFactory,
            optimizer: optim.Optimizer,
            lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
            samplers: List[AbstractSampler],
    ) -> IMetrics:
        diffusion_model.to(device)
        metrics = IMetrics({})
        for epoch in range(start_epoch, self.training_config['epochs']):
            diffusion_model.train()
            start_time = time.time()
            early_stopped, metrics = self._epoch_step(
                diffusion_model=diffusion_model,
                dp=dp,
                split=SPLIT.TRAIN,
                dataset_wrapper=dataset_wrapper,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                writer=writer,
                device=device,
                epoch=epoch,
                samplers=samplers
            )
            self.logger.info(TEXTS.EPOCH_TIME(epoch, time.time() - start_time))
            if early_stopped:
                break
            self._lr_scheduler_step(lr_scheduler, STEP_TIMING.EPOCH, writer)

        return metrics

    def _epoch_step(
            self,
            diffusion_model: AbstractDiffusionModel,
            dp: DiffusionProcess,
            split: SPLIT,
            dataset_wrapper: DiffusionDatasetFactory,
            optimizer: optim.Optimizer,
            lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
            writer: SummaryWriter,
            device: torch.device,
            epoch: int,
            samplers: List[AbstractSampler],
    ) -> Tuple[IEarlyStopped, IMetrics]:
        data_loader = self._get_dataset_loader(split)
        metrics: IMetrics = IMetrics({})
        loss_fn = self.get_loss_fn()

        desc = lambda _batch, _loss: ' | '.join([
            'train' if split == SPLIT.TRAIN else 'eval',
            f'Epoch {epoch + 1} / {self.training_config["epochs"]}',
            f'Batch: {_batch + 1} / {len(data_loader)}',
            f'Total Steps: {self.total_steps + 1}',
            f'loss: {_loss:.4f}' if _loss else ''
        ])

        pbar = tqdm(data_loader, desc=desc(0, None), disable=not self.should_print_to_screen)
        for batch, data in enumerate(pbar):
            x_0, t = data
            x_0, t = x_0.to(device), t.to(device)  # noqa
            optimizer.zero_grad()
            x_t, epsilon = dp.sample(x_0, t)
            epsilon_hat = diffusion_model(x_t, t)
            loss = loss_fn(epsilon, epsilon_hat)
            pbar.set_description(desc(batch + 1, loss.item()))
            if split == SPLIT.TRAIN:
                loss.backward()

                # Gradient clipping
                gradient_clip_value = self.training_config['gradient_clip_value']
                if gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), gradient_clip_value)

                optimizer.step()
                self._lr_scheduler_step(lr_scheduler, STEP_TIMING.BATCH, writer)

                self.total_steps += 1

                self._log_train_metrics(loss.item(), writer, epoch, batch, len(data_loader))
                self._save_checkpoint_step(diffusion_model, optimizer, epoch)

                if (
                        split == SPLIT.TRAIN
                        and self.total_steps >= C_STEPS.WARMUP_STEPS
                        and self.total_steps % C_STEPS.EVAL_STEP == 0
                ):
                    with torch.no_grad():
                        diffusion_model.eval()
                        _, metrics = self._epoch_step(
                            diffusion_model=diffusion_model,
                            dp=dp,
                            split=SPLIT.EVAL,
                            dataset_wrapper=dataset_wrapper,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            writer=writer,
                            device=device,
                            epoch=epoch,
                            samplers=samplers
                        )
                        diffusion_model.train()

                    metrics = self._aggregate_metrics(metrics, device)
                    self._lr_scheduler_step(lr_scheduler, STEP_TIMING.EVALUATION, writer, loss=metrics[METRICS.LOSS])
                    self._log_eval_metrics(metrics, writer)
                    self._sampling_step(
                        model=diffusion_model,
                        samplers=samplers,
                        n_samples=self.samplers_config['num_samples'],
                        dataset_wrapper=dataset_wrapper,
                        writer=writer
                    )

                    if self._early_stopping(metrics[METRICS.LOSS]):
                        return IEarlyStopped(True), metrics
            elif split == SPLIT.EVAL:
                if METRICS.LOSS not in metrics:
                    metrics[METRICS.LOSS.value] = 0
                metrics[METRICS.LOSS.value] += loss.item()
            else:
                raise ValueError(f"Invalid split: {split}")

        if split == SPLIT.EVAL:
            metrics = IMetrics({
                key: value / len(data_loader)
                for key, value in metrics.items()
            })

            metrics = self._aggregate_metrics(metrics, device)

        return IEarlyStopped(False), metrics
