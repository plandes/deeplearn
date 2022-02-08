"""Contains a simple but effective observer pattern set of classes for
training, testing and validating models.

"""
__author__ = 'Paul Landes'

from typing import List, Any, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from zensols.introspect import ClassImporter

mod_logger = logging.getLogger(__name__ + '.status')
"""Logger for this module."""

event_logger = logging.getLogger(__name__ + '.event')
"""Event logger for the :class:`.LogModelObserver."""


class ModelObserver(ABC):
    """Recipient of notifications by the model framework.

    """
    @abstractmethod
    def notify(self, event: str, caller: Any, context: Any = None):
        """Notify all registered observers of an event.

        :param event: the unique identifier of the event using underscore
                      spacing and prefixed by a unique identifier per caller

        :param caller: the object calling this method

        :param context: any object specific to the call and understood by the
                        client on a per client basis

        """
        pass


@dataclass
class FilterModelObserver(ModelObserver):
    """Filters messages from the client to a delegate observer.

    """
    delegate: ModelObserver = field()
    """The delegate observer to notify on notifications from this observer."""

    include_events: Set[str] = field(default_factory=set)
    """A set of events used to indicate to notify :obj:`delegate`."""

    def notify(self, event: str, caller: Any, context: Any = None):
        if event in self.include_events:
            self.delegate(event, caller, context)


@dataclass
class LogModelObserver(ModelObserver):
    """Logs notifications to :mod:`logging` system.

    """
    logger: logging.Logger = field(default=event_logger)
    """The logger that receives notifications."""

    level: int = field(default=logging.INFO)
    """The level used for logging."""

    add_context_format: str = field(default='{event}: {context}')
    """If not ``None``, use the string to format the log message."""

    def notify(self, event: str, caller: Any, context: Any = None):
        if self.logger.isEnabledFor(self.level):
            if self.add_context_format is not None and context is not None:
                event = self.add_context_format.format(
                    **{'event': event, 'context': context})
            self.logger.log(self.level, event)


@dataclass
class RecorderObserver(ModelObserver):
    """Records notifications and provides them as output.

    """
    events: List[Tuple[datetime, str, Any, Any]] = field(default_factory=list)
    """All events received by this observer thus far."""

    flatten: bool = field(default=True)
    """Whether or not make the caller and context in to a strings before storing
    them in :obj:`events`.

    """
    def notify(self, event: str, caller: Any, context: Any = None):
        now = datetime.now()
        if self.flatten:
            caller = ClassImporter.full_classname(caller.__class__)
            if not isinstance(context, (str, bool, int, float)):
                context = str(context)
        self.events.append((now, event, caller, context))

    def events_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.events, columns='time event caller context'.split())


@dataclass
class DumperObserver(RecorderObserver):
    """A class that dumps all data when certain events are received as a CSV to the
    file sytsem.

    """
    output_file: Path = field(default=Path('dumper-observer.csv'))
    """The path to where the (flattened data) is written."""

    trigger_events: Set[str] = field(default_factory=set)
    """A set of all events received that trigger a dump."""

    trigger_callers: Set[str] = field(default=None)
    """A set of all callers' *fully qualified* class names.  If set to ``None`` the
    caller is not a constraint that precludes the dump.

    """
    mkdir: bool = field(default=True)
    """If ``True`` then create the parent directories if they don't exist."""

    def notify(self, event: str, caller: Any, context: Any = None):
        super().notify(event, caller)
        if event in self.trigger_events:
            if self.trigger_callers is not None:
                caller = ClassImporter.full_classname(caller.__class__)
                if caller in self.trigger_callers:
                    return
            df: pd.DataFrame = self.events_as_df()
            if self.mkdir:
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_file)
            if mod_logger.isEnabledFor(logging.INFO):
                mod_logger.error(f'wrote events: {self.output_file}')


@dataclass
class ModelObserverManager(object):
    observers: List[ModelObserver] = field(default_factory=list)
    """A list of observers that get notified of all model lifecycle and process
    events.

    """
    def add(self, observer: ModelObserver):
        """Add an observer to be notified of event changes.

        """
        self.observers.append(observer)

    def notify(self, event: str, caller: Any, context: Any = None):
        """Notify all registered observers of an event.

        :param event: the unique identifier of the event using underscore
                      spacing and prefixed by a unique identifier per caller

        :param caller: the object calling this method

        :param context: any object specific to the call and understood by the
                        client on a per client basis

        """
        for obs in self.observers:
            obs.notify(event, caller, context)
