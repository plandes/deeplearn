# Model

Please first read the [data pre-processing] documentation before proceeding.
In this document, the model for the [Iris example] is defined and we train and
test the model.

The model contains of two kinds of configuration and set up:
* network settings: configuration of the neural network itself, such as types
  and numbers of layers
* model settings: configuration of the model, such as the criterion, optimizer,
  and learning rate


## Network Settings

The network settings declares which network modules to use in the model.  The
network settings is a class that contains the configuration that tells the
module how to build itself, and what that module is.  In the [Iris example],
we first need to create the class:
```python
@dataclass
class IrisNetworkSettings(DeepLinearNetworkSettings):
    def get_module_class_name(self) -> str:
        return __name__ + '.IrisNetwork'
```

We only need to extend from [DeepLinearNetworkSettings], which already has all
the configuration we need since our model is a simple linear set of layers.
We'll define that configuration soon.  However, we must override the abstract
method `get_module_class_name`, which tells us what model to create at
train/test time.

The `IrisNetworkSettings` instance will be populated by the [ConfigFactory]
with fields inherited from [DeepLinearNetworkSettings]:
```ini
[net_settings]
class_name = iris.model.IrisNetworkSettings
dropout = 0.1
activation = None
middle_features = eval: [5, 1]
in_features = 4
out_features = 3
proportions = False
repeats = 1
batch_norm_d = None
batch_norm_features = None
```
which will create a deep linear network expecting 4 input features (one for
each row feature of the flower size data), with 5 times the number of features
in the second layer, same number of feature for the third layer, and finally an
output of three features--one for each flower type.  See
[DeepLinearNetworkSettings] for more details.


## Network Model

Finally, we provide an implementation of [BaseNetworkModule], which extends
from `torch.nn.Module`.  The [BaseNetworkModule] class provides additional
debugging and logging convenience methods.  The method `_debug` in the base
class logs as debug to the passed logger but also provides additional
formatting indicating the name of the model, which is taken from `MODULE_NAME`.
For a simple module like this, it might seem unnecessary.  However, this
additional information is crucial in debugging large models.  The
`_shape_debug` method logs the shape of a tensor.
```python
class IrisNetwork(BaseNetworkModule):
    MODULE_NAME = 'iris'

    def __init__(self, net_settings: IrisNetworkSettings):
        super().__init__(net_settings, logger)
        self.fc = DeepLinear(net_settings)

    def deallocate(self):
        super().deallocate()
        self._deallocate_children_modules()

    def _forward(self, batch: Batch) -> Tensor:
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'label shape: {batch.get_labels().shape}, ' +
                        f'{batch.get_labels().dtype}')

        x = batch.get_flower_dimensions()
        self._shape_debug('input', x)

        x = self.fc(x)
        self._shape_debug('linear', x)

        return x
```
Note that we pass our own logger to the base class, which is needed for the
aforementioned logging convenience methods and is set attribute `logger` in the
base class.  Another important difference when extending from
[BaseNetworkModule] is that the `_forward` is overridden instead of `forward`.
The method signature is also different in that it takes a [Batch] object.
However, any number of arguments can be passed.

In the initializer, all we need to do is create a [DeepLinear] module with the
configuration give from the [ConfigFactory] that was used to create the network
settings instance.

The model is configured as follows
```ini
[model_settings]
class_name = zensols.deeplearn.ModelSettings
path = path: ${default:temporary_dir}/model
nominal_labels = False
learning_rate = 0.1
batch_iteration = gpu
epochs = 15
```
which creates a `ModelSettings` class used to create the model that stores the
model in the temporary directory under `model`.  The `nominal_labels` tell the
framework that the class is not an integer nominal index for each class.  this
is because the model outputs a three neuron output for each flower type (see
the [network model](#network-model) section).  The model will train for 15
epochs using a learning rate of 0.1 using the (default) `torch.nn.Adam`
optimizer with the (default) loss function `torch.nn.CrossEntropyLoss`.  See
[ModelSettings] documentation for more information.

The `batch_iteration = gpu` means the entire batch data set will be decoded in
to GPU memory, which for is possible since this is such a small data set.  If
this parameter was set to `cpu`, all batches would be decoded to CPU memory,
then moved to the GPU for each epoch.


<!-- links -->
[data pre-processing]: preprocess.md
[Iris example]: https://github.com/plandes/deeplearn/blob/master/test/python/iris/model.py
[DeepLinearNetworkSettings]: ../api/zensols.deeplearn.layer.html#zensols.deeplearn.layer.linear.DeepLinearNetworkSettings
[BaseNetworkModule]: ../api/zensols.deeplearn.model.html#zensols.deeplearn.model.module.BaseNetworkModule
[Batch]: ../api/zensols.deeplearn.batch.html#zensols.deeplearn.batch.domain.Batch
[DeepLinear]: ../api/zensols.deeplearn.layer.html#zensols.deeplearn.layer.linear.DeepLinear
[ConfigFactory]: https://plandes.github.io/util/api/zensols.config.html#zensols.config.factory.ConfigFactory
[ModelSettings]: ../api/zensols.deeplearn.html#zensols.deeplearn.domain.ModelSettings
