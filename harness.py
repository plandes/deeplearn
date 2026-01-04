#!/usr/bin/env python

if (__name__ == '__main__'):
    from zensols.cli import ConfigurationImporterCliHarness
    from zensols.deeplearn import TorchConfig
    TorchConfig.init()
    harness = ConfigurationImporterCliHarness(
        src_dir_name='src',
        app_factory_class='zensols.deeplearn.ApplicationFactory',
        proto_args='proto',
        proto_factory_kwargs={'reload_pattern': r'^zensols.deeplearn'},
    )
    harness.run()
