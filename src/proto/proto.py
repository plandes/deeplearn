import logging
import importlib
from zensols.actioncli.config import ExtendedInterpolationConfig

logger = logging.getLogger(__name__)


def create_config():
    return ExtendedInterpolationConfig('resources/dltools.conf')


def tmp():
    import zensols.dltools.app
    importlib.reload(zensols.dltools.app)
    logging.getLogger('zensols.actioncli').setLevel(logging.INFO)
    logging.getLogger('zensols.dltools.app').setLevel(logging.DEBUG)
    app = zensols.dltools.app.MainApplication(create_config())
    app.tmp()


def main():
    logging.basicConfig(level=logging.WARNING)
    run = 1
    {1: tmp,
     }[run]()


main()
