"""Contains a class to assist in the training lifecycle of the
:class:`.ModelExecutor`.

"""
__author__ = 'Paul Landes'

from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import logging
from logging import Logger
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch import nn
from zensols.deeplearn.result import EpochResult


class UpdateAction(Enum):
    """An action type to invoke on the :class:`.ModelExecutor` during training.

    """
    ITERATE_EPOCH = 0
    SET_EPOCH = 1
    STOP = 2


@dataclass
class LifeCycleStatus(object):
    """Indicates what to do in the next epoch of the training cycle.

    """
    action: UpdateAction
    epoch: int = field(default=None)
    reason: str = field(default=None)


@dataclass
class LifeCycleManager(object):
    """The class is used to assist in the training lifecycle of the
    :class:`.ModelExecutor`.  It watches for a file on the file system to
    provide instructions on what to do in the next epoch.

    """
    status_logger: Logger
    progress_logger: Logger
    update_path: Path
    pbar: tqdm = field(default=None)

    def reset(self, optimizer: nn.L1Loss, scheduler: Any,
              pbar: tqdm, n_epochs: int):
        # clear any early stop state
        if self.update_path is not None and self.update_path.is_file():
            self.logger.info(f'cleaning update file: {self.update_path}')
            self.update_path.unlink()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pbar = pbar
        self.n_epochs = n_epochs
        self.current_epoch = 0
        # set initial "min" to infinity
        self.valid_loss_min = np.Inf
        if self.status_logger.isEnabledFor(logging.INFO):
            self.status_logger.info(f'watching update file {self.update_path}')

    def update_loss(self,
                    valid_epoch_result: EpochResult,
                    train_epoch_result: EpochResult,
                    ave_valid_loss: float):
        progress_logger = self.progress_logger
        optimizer = self.optimizer
        scheduler = self.scheduler
        pbar = self.pbar
        valid_loss = valid_epoch_result.ave_loss
        logger = self.status_logger

        # adjust the learning rate if a scheduler is configured
        if scheduler is not None:
            scheduler.step(ave_valid_loss)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('epoch ave valid loss/results averaged valid_loss ' +
                         f'{ave_valid_loss}/{valid_loss}, ' +
                         f'losses: {len(valid_epoch_result.losses)}')

        decreased = valid_loss < self.valid_loss_min
        dec_str = '\\/' if decreased else '/\\'
        if abs(ave_valid_loss - valid_loss) > 1e-10:
            logger.warning('validation loss and result do not match: ' +
                           f'{ave_valid_loss} - {valid_loss} > 1e-10')
        msg = (f'tr:{train_epoch_result.ave_loss:.3f}|' +
               f'va min:{self.valid_loss_min:.3f}|va:{valid_loss:.3f}')
        if scheduler is not None:
            lr = self._get_optimizer_lr(optimizer)
            msg += f'|lr:{lr}'
        msg += f' {dec_str}'

        if pbar is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(msg)
            pbar.set_description(msg)
        else:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'epoch {self.current_epoch}/{self.n_epochs}: {msg}')

        # save model if validation loss has decreased
        if decreased:
            if progress_logger.isEnabledFor(logging.DEBUG):
                progress_logger.debug('validation loss decreased min/iter' +
                                      f'({self.valid_loss_min:.6f}' +
                                      f'/{valid_loss:.6f}); saving model')
            self.valid_loss_min = valid_loss
        else:
            if progress_logger.isEnabledFor(logging.DEBUG):
                progress_logger.debug('validation loss increased min/iter' +
                                      f'({self.valid_loss_min:.6f}' +
                                      f'/{valid_loss:.6f})')

        return self.valid_loss_min, decreased

    def _read_status(self) -> LifeCycleStatus:
        """Read the early stop/update file and return a value to update the current
        epoch number (if any).

        """
        update = LifeCycleStatus(UpdateAction.ITERATE_EPOCH)
        update_path = self.update_path
        if update_path is not None:
            if self.status_logger.isEnabledFor(logging.DEBUG):
                self.status_logger.debug(f'update check at {update_path}')
            if update_path.exists():
                data = None
                try:
                    with open(update_path) as f:
                        data = json.load(f)
                    if 'epoch' in data:
                        epoch = int(data['epoch'])
                        update.epoch = epoch
                        update.action = UpdateAction.SET_EPOCH
                        update.reason = (f'update from {update_path}: ' +
                                         f'setting epoch to: {epoch}')
                except Exception as e:
                    reason = f'bad format in {update_path}--assume exit: {e}'
                    update.action = UpdateAction.STOP
                    update.reason = reason
                update_path.unlink()
        return update

    def get_status(self) -> LifeCycleStatus:
        """Return the epoch to set in the training loop of the :class:`.ModelExecutor`.

        """
        status = self._read_status()
        if status.action == UpdateAction.STOP:
            # setting to the max value fails the executors train outter loop
            # causing a robust non-error exit
            status.epoch = sys.maxsize
        elif status.action == UpdateAction.SET_EPOCH:
            self.current_epoch = status.epoch
            if self.pbar is not None:
                self.pbar.reset()
                self.pbar.update(self.current_epoch)
        elif status.action == UpdateAction.ITERATE_EPOCH:
            self.current_epoch += 1
            status.epoch = self.current_epoch
            if self.pbar is not None:
                self.pbar.update()
            if self.current_epoch >= self.n_epochs:
                status.action = UpdateAction.STOP
                status.reason = f'epoch threshold reached at {self.n_epochs}'
        else:
            raise ValueError(f'unknownn status: {status}')
        if status.reason and self.status_logger.isEnabledFor(logging.INFO):
            self.status_logger.info(status.reason)
        return status

    def stop(self) -> bool:
        """Stops the execution of training the model.

        Currently this is done by creating a file the executor monitors.

        """
        update_path = self.update_path
        if update_path is not None and not update_path.is_file():
            update_path.touch()
            self.status_logger.info(f'created early stop file: {update_path}')
            return True
        return False
