# Results

The [facade] produces results both from the training phase and the testing
phase.  This class is composed of the following as a hierarchical tree like
object graph including:
* [ModelResult]: top level object that contains the results from all data sets
  as properties `test`, `train` and `validation`, which are all properties of
  type [DatasetResult]
* [DatasetResult]: results of data set results given in the [results property]
  as [EpochResult] instances
* [EpochResult]: results for each epoch containing the [labels], [predictions],
  and [metrics].

Either [model type] of the results determines what kind of [metrics] are
provided as either:

* [prediction metrics]: R^2, RMSE, MAE, and correlation
* [classification metrics]: accuracy, micro and macro F1, recall and precision

The [facade] provides access to the [last_result], which has just the training
results if only trained, or both the training and test results after testing as
a [ModelResult].  Note that you must call [persist_result] to store the results
after training and/or testing to get those respective results as detailed in
[facade results] section.


## Result Manager

The [result_manager] property of the [facade] provides an instance of a
[ModelManager].  You can use this instance to retrieve previous results in the
case they were computed in a previous Python interpreter.


## Reproducibility

Being able to reproduce the results is one of the major goals of this
framework.  While this framework provides an API ([TorchConfig]) to set the
[random seed state] of [PyTorch], numpy, and the Python environment, there is
still some variance in some cases in results.

According to this [GitHub issue](https://github.com/pytorch/pytorch/issues/18412):
> This is expected, some of our kernels are not deterministic (specially during backward).
> Might be good to refer to [#1535](https://github.com/pytorch/pytorch/issues/15359).


<!-- links -->

[PyTorch]: https://pytorch.org

[facade]: facade.md
[facade resources]: facade.html#resources

[ModelResult]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.ModelResult
[DatasetResult]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.DatasetResult
[results property]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.DatasetResult.results
[EpochResult]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.EpochResult
[labels]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.ResultsContainer.labels
[predictions]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.ResultsContainer.predictions
[metrics]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.ResultsContainer.metrics
[model type]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.ResultsContainer.model_type
[prediction metrics]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.ResultsContainer.prediction_metrics
[classification metrics]: ../api/zensols.deeplearn.result.html#zensols.deeplearn.result.domain.ResultsContainer.classification_metrics
[ModelManager]: ../api/zensols.deeplearn.model.html#zensols.deeplearn.model.manager.ModelManager
[last_result]: ../api/zensols.deeplearn.model.html#zensols.deeplearn.model.facade.ModelFacade.last_result
[persist_result]: ../api/zensols.deeplearn.model.html#zensols.deeplearn.model.facade.ModelFacade.persist_result

[TorchConfig]: ../api/zensols.deeplearn.html#zensols.deeplearn.torchconfig.TorchConfig
[random seed state]: ../api/zensols.deeplearn.html#zensols.deeplearn.torchconfig.TorchConfig.set_random_seed
[result_manager]: ../api/zensols.deeplearn.model.html#zensols.deeplearn.model.executor.ModelExecutor.result_manager
