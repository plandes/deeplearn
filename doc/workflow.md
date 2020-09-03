# Workflow

This package provides a workflow for processing features, training and then
testing a model.  A high level outline of this process follows:
1. Define a [Stash] to generate data points to be vectorized.
1. Vectorize features using a [FeatureVectorizer].
1. Store the vectorized features to disk so they can be retrieved quickly and
   frequently.
1. At train time, load the vectorized features in to memory and train.
1. Test the model and store the results to disk.

The [Iris example] (also see the [Iris example configuration]) is the most
basic example of how to use this framework and will be explored in more depth
in this document.


## Pre-Processing Data

Processing the raw data in your application's data set to creation of the
batches is a pre-process set that happens before the model is trained.
Everything in [PyTorch] is done with tensors, so we first must be able to
process the data in to some numeric form.  The framework uses the [Stash]
instances since they allow an easy way to index all data points with a unique
key.  After the data is processed across each instance, all data is then
vectorized in to *batches*, which contain a grouping of data and used as
mini-batches.


### Configuration File

Please first read the [configuration factory] documentation as they are tightly
integrated with this framework.  We'll start our configuration file with
defining locations of the data set, where to store temporary and result files:
```ini
[default]
root_dir = ${env:app_root}
test_resources = ${root_dir}/test-resources/iris
temporary_dir = ${root_dir}/target/iris
temporary_batch_dir = ${temporary_dir}/batch
results_dir = ${temporary_dir}/results
```

Next we'll add two instances of a [TorchConfig], which help with allocating
[PyTorch] resources:
```ini
[torch_config]
class_name = zensols.deeplearn.TorchConfig
use_gpu = False
data_type = eval({'import': ['torch']}): torch.float32

[gpu_torch_config]
class_name = zensols.deeplearn.TorchConfig
use_gpu = True
data_type = eval({'import': ['torch']}): torch.float32
```
which define a CPU based configuration, which will be used for creating
batches.  The GPU based configuration will be used for creating tensors in the
GPU device.  Both default to creating tensors of type 32-bit floats.


### Processing Data Points

Each data observation is called a *data point* in the framework and extends
from [DataPoint].  Your application must extend this class and define
properties and attributes that access the data that is to be vectorized.

The [Iris example] dataset comes as a CSV file, so we can use the framework's
[Pandas] class [DefaultDataframeStash], which we can use to create instances of
[DataPoint] directly:
```ini
[dataset_stash]
class_name = zensols.dataframe.DefaultDataframeStash
dataframe_path = path: ${default:temporary_dir}/df.dat
key_path = path: ${default:temporary_dir}/keys.dat
split_col = ds_type
input_csv_path = path: ${default:test_resources}/iris.csv
```

This creates an [Stash] instance that pre-processes data from a CSV file found
at `input_csv_path` in a ready to use format and pickles it in a file at
`dataframe_path`, which we've defined to be in our temporary file space.  If
the directory doesn't exist, it will create it.

The [DefaultDataframeStash] needs a column to indicate to which data set the
respective point belongs.  A column called `ds_type` was added to [Iris data
set] in this repository for this reason.  The `split_col` is given this column
to create a set of keys for each data set for fast retrieval and access.  The
key splits (list of keys for each split) are pickled in the `key_path`

Next we create a [Stash] that create a new and separate stash for each data
set:
```ini
[dataset_split_stash]
class_name = zensols.dataset.DatasetSplitStash
delegate = instance: dataset_stash
split_container = instance: dataset_stash
```

The [DatasetSplitStash] has a [splits method] that returns a dictionary of
string split name to data set based on the key splits we defined earlier.  In
this case, we set both the `delegate`, which is the stash to use the data, and
the `split_container` as the [SplitStashContainer] since
[DefaultDataframeStash] serves both these purposes.  Note that for non-data
frame containers, this step of defining the resources need to be configured
with care.

For the Iris example, the [IrisDataPoint] extends the [DataPoint] and contains
a [Pandas] row:
```python
@dataclass
class IrisDataPoint(DataPoint):
    LABEL_COL = 'species'
    FLOWER_DIMS = 'sepal_length sepal_width petal_length petal_width'.split()

    row: pd.Series

    @property
    def label(self) -> str:
        return self.row[self.LABEL_COL]

    @property
    def flower_dims(self) -> pd.Series:
        return [self.row[self.FLOWER_DIMS]]
```


<!-- links -->
[PyTorch]: https://pytorch.org
[Pandas]: https://pandas.pydata.org
[Iris data set]: https://archive.ics.uci.edu/ml/datasets/iris
[configuration factory]: https://plandes.github.io/util/doc/config.html#configuration-factory

[Iris example configuration]: https://github.com/plandes/deeplearn/blob/master/test-resources/iris
[Iris example]: https://github.com/plandes/deeplearn/blob/master/test/python/iris/model.py
[IrisDataPoint]: https://github.com/plandes/deeplearn/blob/master/test/python/iris/model.py

[Stash]: https://plandes.github.io/util/api/zensols.persist.html#zensols.persist.domain.Stash
[FeatureVectorizer]: ../api/zensols.deeplearn.vectorize.html#zensols.deeplearn.vectorize.domain.FeatureVectorizer
[DataPoint]: ../api/zensols.deeplearn.batch.html#zensols.deeplearn.batch.domain.DataPoint
[TorchConfig]: ../api/zensols.deeplearn.html#zensols.deeplearn.torchconfig.TorchConfig
[DefaultDataframeStash]: ../api/zensols.dataframe.html#zensols.dataframe.stash.DefaultDataframeStash
[DatasetSplitStash]: ../api/zensols.dataset.html#zensols.dataset.stash.DatasetSplitStash
[splits method]: ../api/zensols.dataset.html?highlight=datasetsplitstash#zensols.dataset.interface.SplitStashContainer.splits
[SplitStashContainer]: ../api/zensols.dataset.html#zensols.dataset.interface.SplitStashContainer
