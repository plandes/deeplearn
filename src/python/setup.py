from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.deeplearn",
    package_names=['zensols', 'resources'],
    description='General deep learing utility library',
    user='plandes',
    project='deeplearn',
    keywords=['tooling'],
    has_entry_points=False,
).setup()
