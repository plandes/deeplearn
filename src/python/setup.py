from pathlib import Path
from zensols.pybuild import SetupUtil

SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.deeplearn",
    package_names=['zensols', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    description='General deep learing utility library',
    user='plandes',
    project='deeplearn',
    keywords=['tooling'],
    has_entry_points=False,
).setup()
