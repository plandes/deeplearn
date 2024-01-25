"""Contains a class to assist in the training of the :class:`.ModelExecutor`.

"""
__author__ = 'Paul Landes'

from typing import Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys
import logging
from logging import Logger
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from zensols.deeplearn import ModelError
from zensols.deeplearn.result import EpochResult


class UpdateAction(Enum):
    """An action type to invoke on the :class:`.ModelExecutor` during training.

    """
    ITERATE_EPOCH = 0
    SET_EPOCH = 1
    STOP = 2


@dataclass
class TrainStatus(object):
    """Indicates what to do in the next epoch of the training cycle.

    """
    action: UpdateAction
    epoch: int = field(default=None)
    reason: str = field(default=None)


@dataclass
class TrainManager(object):
    """The class is used to assist in the training of the
    :class:`.ModelExecutor`.  It updates validation loss and helps with early
    stopping decisions.  It also watches for a file on the file system to
    provide instructions on what to do in the next epoch.

    """
    status_logger: Logger = field()
    """The logger to record status updates during training."""

    progress_logger: Logger = field()
    """The logger to record progress updates during training.  This is used only
    when the progress bar is turned off (see
    :meth:`.ModelFacade._configure_cli_logging`).

    """
    update_path: Path = field()
    """See :obj:`.ModelExecutor.update_path`.

    """
    max_consecutive_increased_count: int = field()
    """See :obj:`.Domain.max_consecutive_increased_count`.

    """
    progress_bar_number_width: int = field(default=6)
    """The string width of the train/validation loss metrics in the progress
    bar, which needs to be greater than 4.

    """
    def start(self, optimizer: nn.L1Loss, scheduler: Any,
              n_epochs: int, pbar: tqdm):
        # clear any early stop state
        if self.update_path is not None and self.update_path.is_file():
            self.progress_logger.info(
                f'cleaning update file: {self.update_path}')
            self.update_path.unlink()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.current_epoch = 0
        self.consecutive_increased_count = 0
        # set initial "min" to infinity
        self.valid_loss_min = np.Inf
        self.pbar = pbar
        if self.progress_logger.isEnabledFor(logging.INFO):
            self.progress_logger.info(
                f'watching update file {self.update_path}')
        self.validation_loss_decreases = 0

    def _get_optimizer_lr(self, optimizer: torch.optim.Optimizer) -> float:
        """Return the current optimizer learning rate, which can be modified by
        a scheduler if one is configured.

        """
        param_group = next(iter(optimizer.param_groups))
        return float(param_group['lr'])

    def _fixed_sci_format(self, v: str) -> str:
        """Format a number to a width resorting to scientific notation where
        necessary.  The returned string is left padded with space in cases where
        scientific notation is too wide for ``v > 0``.  The mantissa is cut off
        also for ``v > 0`` when the string version of the number is too wide.

        """
        length: int = self.progress_bar_number_width
        n: int = length
        ln: int = None
        pad: int = None
        while n > 0:
            i = len('%#.*g' % (n, v))
            s = '%.*g' % (n + n - i, v)
            ln = len(s)
            pad = length - ln
            if pad >= 0:
                break
            n -= 1
        if pad > 0:
            s = (' ' * pad) + s
        return s

    def update_loss(self, valid_epoch_result: EpochResult,
                    train_epoch_result: EpochResult,
                    ave_valid_loss: float) -> Tuple[float, bool]:
        """Update the training and validation loss.

        :return: a tuple of the latest minimum validation loss and whether or
                 not the last validation loss has decreased

        """
        logger = self.status_logger
        progress_logger = self.progress_logger
        optimizer = self.optimizer
        valid_loss = valid_epoch_result.ave_loss
        sfmt = self._fixed_sci_format

        # adjust the learning rate if a scheduler is configured
        if self.scheduler is not None:
            # the LambdaLR scheduler creates warnings when it gets the
            # validation loss
            if isinstance(self.scheduler, LambdaLR):
                self.scheduler.step()
            else:
                self.scheduler.step(ave_valid_loss)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('epoch ave valid loss/results averaged valid_loss ' +
                         f'{ave_valid_loss}/{valid_loss}, ' +
                         f'losses: {len(valid_epoch_result.losses)}')

        decreased = valid_loss < self.valid_loss_min
        dec_str = '\\/' if decreased else '/\\'
        if abs(ave_valid_loss - valid_loss) > 1e-10:
            logger.warning('validation loss and result are not close: ' +
                           f'{ave_valid_loss} - {valid_loss} > 1e-10')
        if train_epoch_result.contains_results:
            train_loss = train_epoch_result.ave_loss
        else:
            train_loss = -1
        msg = (f'tr:{sfmt(train_loss)}|' +
               f'va min:{sfmt(self.valid_loss_min)}|' +
               f'va:{sfmt(valid_loss)}')
        if self.scheduler is not None:
            lr = self._get_optimizer_lr(optimizer)
            msg += f'|lr:{sfmt(lr)}'
        msg += f' {dec_str}'

        if self.pbar is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(msg)
            self.pbar.set_description(msg)
        else:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'epoch {self.current_epoch}/' +
                            f'{self.n_epochs}: {msg}')

        # save model if validation loss has decreased
        if decreased:
            if progress_logger.isEnabledFor(logging.DEBUG):
                progress_logger.debug('validation loss decreased min/iter' +
                                      f'({self.valid_loss_min:.6f}' +
                                      f'/{valid_loss:.6f}); saving model')
            self.valid_loss_min = valid_loss
            self.consecutive_increased_count = 0
            self.validation_loss_decreases += 1
        else:
            if progress_logger.isEnabledFor(logging.DEBUG):
                progress_logger.debug('validation loss increased min/iter' +
                                      f'({self.valid_loss_min:.6f}' +
                                      f'/{valid_loss:.6f})')
            self.consecutive_increased_count += 1

        return self.valid_loss_min, decreased

    def _read_status(self) -> TrainStatus:
        """Read the early stop/update file and return a value to update the
        current epoch number (if any).

        """
        update = TrainStatus(UpdateAction.ITERATE_EPOCH)
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

    def _get_stop_reason(self) -> str:
        reason = None
        if self.current_epoch >= self.n_epochs:
            reason = f'epoch threshold reached at {self.n_epochs}'
        elif (self.consecutive_increased_count >
              self.max_consecutive_increased_count):
            reason = ('reached max consecutive increased count: ' +
                      f'{self.max_consecutive_increased_count}')
        return reason

    def get_status(self) -> TrainStatus:
        """Return the epoch to set in the training loop of the
        :class:`.ModelExecutor`.

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
            stop_reason = self._get_stop_reason()
            if self.pbar is not None:
                self.pbar.update()
            if stop_reason is not None:
                status.action = UpdateAction.STOP
                status.reason = stop_reason
        else:
            raise ModelError(f'Unknownn status: {status}')
        if status.reason and self.status_logger.isEnabledFor(logging.INFO):
            self.status_logger.info(status.reason)
        return status

    def stop(self) -> bool:
        """Stops the execution of training the model.  Currently this is done by
        creating a file the executor monitors.

        :return: ``True`` if the application is configured to early stop and
                 the signal has not already been given

        """
        update_path = self.update_path
        if update_path is not None and not update_path.is_file():
            update_path.parent.mkdir(parents=True, exist_ok=True)
            update_path.touch()
            self.status_logger.info(f'created early stop file: {update_path}')
            return True
        return False
