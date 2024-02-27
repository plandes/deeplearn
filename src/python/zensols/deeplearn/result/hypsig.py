"""Model hypothesis significance testing.  This module has a small framework for
the hypothesis testing the model results (typically the results from the test
dataset).  The outcome of disproving the null hypothesis (which is that two
classifiers perform the same) means that a classifier has statistically
significant better (or worse) performance compared to a second.

"""
__author__ = 'Paul Landes'

from typing import (
    Set, Tuple, List, Sequence, Dict, Any, Iterable, Type, ClassVar
)
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import sys
import logging
import math
from itertools import chain
from io import TextIOBase
import numpy as np
import pandas as pd
from zensols.util import APIError
from zensols.persist import persisted
from zensols.deeplearn.dataframe import DataFrameDictable
from zensols.datdesc import DataFrameDescriber

logger = logging.getLogger(__name__)


class SignificanceError(APIError):
    """Raised for inconsistent or bad data while testing significance."""
    pass


@dataclass
class Evaluation(DataFrameDictable):
    """An evaluation metric returned by an implementation of
    :class:`.SignificanceTest`.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = frozenset({'disprove_null_hyp'})

    pvalue: float = field()
    """The probabily value (p-value)."""

    alpha: float = field()
    """Independency threshold for asserting the null hypothesis."""

    statistic: float = field(default=None)
    """A method specific statistic."""

    @property
    def disprove_null_hyp(self) -> bool:
        """Whether the evaluation shows the test disproves the null hypothesis.

        """
        return self.pvalue < self.alpha

    def _write_key_value(self, k: Any, v: Any, depth: int, writer: TextIOBase):
        if isinstance(v, float):
            v = f'{v:e}'
        self._write_line(f'{k}: {v}', depth, writer)


@dataclass
class SignificanceTestData(DataFrameDictable):
    """Metadata needed to create significance tests.

    :see: :class:`.SignificanceTest`.

    """
    a: pd.DataFrame = field()
    """Test set results from the first model."""

    b: pd.DataFrame = field()
    """Test set results from the second model."""

    id_col: str = field(default='id')
    """The dataset column that contains the unique identifier of the data point.
    If this is not ``None``, an assertion on the id's of :obj:`a` and :obj:`b`
    is performed.

    """
    gold_col: str = field(default='label')
    """The column of the gold label/data."""

    pred_col: str = field(default='pred')
    """The column of the prediction."""

    alpha: float = field(default=0.05)
    """Used to compare with the p-value to disprove the null hypothesis."""

    null_hypothesis: str = field(default=(
        'classifiers have a similar proportion of errors on the test set'))
    """A human readable string of the hypothesis."""

    def _assert_data(self):
        dfa: pd.DataFrame = self.a
        dfb: pd.DataFrame = self.b
        assert len(dfa) == len(dfb)
        if self.id_col is not None:
            if dfa[self.id_col].tolist() != dfb[self.id_col].tolist():
                raise SignificanceError(
                    f"Test result IDs do not match for column '{self.id_col}'")
        if dfa[self.gold_col].tolist() != dfb[self.gold_col].tolist():
            raise SignificanceError(
                f"Test result labels do not match for column '{self.gold_col}'")

    @property
    @persisted('_correct_table')
    def correct_table(self) -> pd.DataFrame:
        """Return a tuple of a dataframe of the correct values in columns
        ``a_correct`` and ``b_correct``.

        """
        dfa: pd.DataFrame = self.a
        dfb: pd.DataFrame = self.b
        # each classifier's correct classification by ID
        return pd.concat(
            (dfa[self.id_col],
             dfa[self.gold_col] == dfa[self.pred_col],
             dfb[self.gold_col] == dfb[self.pred_col]),
            axis=1, keys='id a_correct b_correct'.split())

    @property
    @persisted('_contingency_table')
    def contingency_table(self) -> pd.DataFrame:
        """Return the contingency table using correct columns from
        :obj:`correct_table``.

        """
        dfc: pd.DataFrame = self.correct_table
        df_cont: pd.DataFrame = pd.crosstab(
            dfc['a_correct'], dfc['b_correct'],
            rownames='a'.split(),
            colnames='b'.split())
        return df_cont


# subclass this into a regression/ranking from classification when needed
@dataclass
class SignificanceTest(DataFrameDictable, metaclass=ABCMeta):
    """A statistical significance hypothesis test for models using test set data
    results.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = frozenset({'evaluation'})

    data: SignificanceTestData = field()
    """Contains the data to be used for the significance hypothesis testing."""

    @property
    def name(self) -> str:
        """The name of the test."""
        return self._NAME

    @abstractmethod
    def _compute_significance(self, data: SignificanceTestData) -> Evaluation:
        """Compute the significance of the result ``data``."""
        pass

    @property
    @persisted('_evaluation')
    def evaluation(self) -> Evaluation:
        self.data._assert_data()
        return self._compute_significance(self.data)

    def write_conclusion(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        """Write an intuitive explanation of the results.

        :param depth: the starting indentation depth

        :param writer: the writer to dump the content of this writable

        """
        data: SignificanceTestData = self.data
        res: Evaluation = self.evaluation
        disprove_null_hyp: bool = res.disprove_null_hyp
        disprove_str: str = 'disproved' \
            if disprove_null_hyp else 'did not disprove'
        self._write_line(f'{disprove_str} the null hypothesis:', depth, writer)
        self._write_line(f"'{data.null_hypothesis}' is {not disprove_null_hyp}",
                         depth, writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_contingency: bool = True,
              include_conclusion: bool = True):
        if include_contingency:
            self._write_line('contingency:', depth, writer)
            self._write_dataframe(self.data.contingency_table,
                                  depth + 1, writer)
        self._write_line('evaluation:', depth, writer)
        self._write_object(self.evaluation, depth + 1, writer)
        if include_conclusion:
            self._write_line('hypothesis:', depth, writer)
            self.write_conclusion(depth + 1, writer)


@dataclass
class SignificanceTestSuite(DataFrameDictable):
    """A suite of significance tests that use one or more
    :class:`.SignificanceTest`.

    """
    _TESTS: ClassVar[Dict[str, Type[SignificanceTest]]] = {}
    """A mapping of all available significance tests."""

    data: SignificanceTestData = field()
    """Contains the data to be used for the significance hypothesis testing."""

    test_names: Tuple[str, ...] = field(default=None)
    """The test names (:obj:`.SignificanceTest.name`) to be in this suite."""

    @classmethod
    def _register_test(cls: Type, test: Type[SignificanceTest]):
        cls._TESTS[test._NAME] = test

    @property
    def available_test_names(self) -> Set[str]:
        """All avilable names of tests (see :obj:`test_names`)."""
        return set(self._TESTS.keys())

    @property
    def tests(self) -> Tuple[SignificanceTest, ...]:
        """The tests used in this suite"""
        def map_test_name(name: str) -> SignificanceTest:
            cls: str = self._TESTS[name]
            return cls(data=self.data)

        test_names: Sequence[str, ...] = self.test_names
        if test_names is None:
            test_names = sorted(self.available_test_names)
        return tuple(map(map_test_name, test_names))

    @property
    def describer(self) -> DataFrameDescriber:
        """A dataframe describer of all significance evaluations."""
        rows: List[Tuple[Any, ...]] = []
        test: SignificanceTest
        for test in self.tests:
            evl: Evaluation = test.evaluation
            rows.append((test.name, evl.pvalue, evl.statistic,
                         evl.disprove_null_hyp))
        return DataFrameDescriber(
            name='significance-tests',
            df=pd.DataFrame(rows, columns='name pvalue stat disprove'.split()),
            desc='Model Result Significance Tests',
            meta=(('name', 'the name of the significance test'),
                  ('pvalue', "the test's resulting p-value"),
                  ('stat', "the test's resulting statistic"),
                  ('disprove', 'if true, the null hypothesis is disproven')))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('contingency:', depth, writer)
        self._write_dataframe(self.data.contingency_table, depth + 1, writer)
        test: SignificanceTest
        for test in self.tests:
            self._write_line(f'{test.name}:', depth, writer)
            test.write(depth + 1, writer, include_contingency=False)


class StudentTTestSignificanceTest(SignificanceTest):
    """Student's T-Test, which measure the difference in the mean.  This test
    violates the independence assumption, but it is included as it is still used
    in papers as a metric.

    Citation:

      `Student (1908)`_ The Probable Error of a Mean. Biometrika, 6(1):1–25.

    .. _Student (1908): https://www.jstor.org/stable/2331554


    """
    _NAME: ClassVar[str] = 'student-ttest'

    def _compute_significance(self, data: SignificanceTestData) -> Evaluation:
        from scipy.stats import ttest_ind
        dfc = data.correct_table
        res = ttest_ind(dfc['a_correct'], dfc['b_correct'])
        return Evaluation(
            pvalue=res.pvalue,
            alpha=data.alpha,
            statistic=res.statistic)


SignificanceTestSuite._register_test(StudentTTestSignificanceTest)


class AnovaSignificanceTest(SignificanceTest):
    """One-way ANOVA test."""
    _NAME: ClassVar[str] = 'anova'

    def _compute_significance(self, data: SignificanceTestData) -> Evaluation:
        from scipy.stats import f_oneway
        dfc = data.correct_table
        stat, pvalue = f_oneway(dfc['a_correct'], dfc['b_correct'])
        return Evaluation(
            pvalue=pvalue,
            alpha=data.alpha,
            statistic=stat)


SignificanceTestSuite._register_test(AnovaSignificanceTest)


class WilcoxSignificanceTest(SignificanceTest):
    """Wilcoxon signed-rank test, which is a non-parametric version of Student's
    T-Test.

    Citation:

      `Frank Wilcoxon (1945)`_ Individual Comparisons by Ranking
      Methods. Biometrics Bulletin, 1(6):80–83.

    .. _Frank Wilcoxon (1945): https://www.jstor.org/stable/3001968

    """
    _NAME: ClassVar[str] = 'wilcoxon'

    def _compute_significance(self, data: SignificanceTestData) -> Evaluation:
        from scipy.stats import wilcoxon
        dfc = data.correct_table
        a = dfc['a_correct'].apply(lambda x: 1 if x else 0)
        b = dfc['b_correct'].apply(lambda x: 1 if x else 0)
        res = wilcoxon(a, b)
        return Evaluation(
            pvalue=res.pvalue,
            alpha=data.alpha,
            statistic=res.statistic)


SignificanceTestSuite._register_test(WilcoxSignificanceTest)


class McNemarSignificanceTest(SignificanceTest):
    """McNemar's test.

    Citation:

      `Quinn McNemar (1947)`_ Note on the sampling error of the difference
      between correlated proportions or percentages. Psychometrika,
      12(2):153–157, June.

    .. _Quinn McNemar (1947): https://doi.org/10.1007/BF02295996

    """
    _NAME: ClassVar[str] = 'mcnemar'

    def _compute_significance(self, data: SignificanceTestData) -> Evaluation:
        from statsmodels.stats.contingency_tables import mcnemar
        dfc: pd.DataFrame = data.correct_table
        df_cont: pd.DataFrame = data.contingency_table
        # Yes/No is the count of test instances that Classifier1 got correct and
        # Classifier2 got incorrect, and No/Yes is the count of test instances
        # that Classifier1 got incorrect and Classifier2 got correct
        yes_no = df_cont.loc[True][False]
        no_yes = df_cont.loc[False][True]
        assert yes_no == len(dfc[dfc['a_correct'] & ~dfc['b_correct']])
        assert no_yes == len(dfc[~dfc['a_correct'] & dfc['b_correct']])
        # compute stat and pvalue
        res = mcnemar(df_cont, exact=False, correction=True)
        return Evaluation(res.pvalue, data.alpha, res.statistic)


SignificanceTestSuite._register_test(McNemarSignificanceTest)


@dataclass
class ChiSquareEvaluation(Evaluation):
    """The statistics gathered from :func:`scipy.stats.chi2_contingency` and
    created in :class:`.ChiSquareCalculator`.

    """
    dof: int = field(default=None)
    """Degrees of freedom"""

    expected: np.ndarray = field(default=None)
    """The expected frequencies, based on the marginal sums of the table.  It
    has the same shape as :class:`.ChiSquareCalculator`.observations.

    """
    contingency_table: pd.DataFrame = field(default=None)
    """The contigency table used for the results."""

    @property
    def associated(self) -> bool:
        """Whether or not the variables are assocated (rejection of the null
        hypotheis).

        """
        return self.pvalue <= self.alpha

    @property
    def raw_residuals(self) -> pd.DataFrame:
        """The raw residuals as computed as the difference between the
        observations and the expected cell values.

        """
        return self.contingency_table - self.expected

    @property
    def contribs(self) -> pd.DataFrame:
        """The contribution of each cell to the results of the chi-square
        computation.

        """
        return self.pearson_residuals ** 2

    @property
    def pearson_residuals(self) -> pd.DataFrame:
        """Pearson residuals, aka *standardized* residuals."""
        exp = self.expected
        raw_resid = self.contingency_table - exp
        return raw_resid / np.sqrt(exp)

    @property
    def adjusted_residuals(self) -> pd.DataFrame:
        """The adjusted residuals (see class docs)."""
        obs_df = self.contingency_table
        obs = obs_df.to_numpy()
        exp = self.expected.to_numpy()
        raw_res = self.raw_residuals.to_numpy()
        row_marg = obs.sum(axis=0)
        col_marg = obs.sum(axis=1)
        n = obs.sum()
        arr = np.empty(shape=exp.shape)
        for rix in range(exp.shape[0]):
            for cix in range(exp.shape[1]):
                rm = row_marg[cix]
                cm = col_marg[rix]
                num = raw_res[rix][cix]
                ex = exp[rix][cix]
                mul = ((1. - (rm / n)) * (1. - (cm / n)))
                to_sqrt = ex * mul
                if to_sqrt == 0:
                    logger.warning(f'bad multiplier: xpected={ex}, ' +
                                   f'rm={rm}, cm={cm}, n={n}, mul={mul}')
                denom = math.sqrt(to_sqrt)
                v = num / denom
                arr[rix][cix] = v
        return pd.DataFrame(arr, columns=obs_df.columns, index=obs_df.index)

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        fs = ('associated expected contribs ' +
              'adjusted_residuals pearson_residuals').split()
        return chain.from_iterable(
            [super()._get_dictable_attributes(), map(lambda x: (x, x), fs)])

    def write_associated(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        """Write how the variables relate as a result of the chi-square
        computation.

        :see: :meth:`write`

        """
        if self.associated:
            assoc = 'variables are associated (reject H0)'
        else:
            assoc = 'variables are not associated (fail to reject H0)'
        self._write_line(f'associated: {assoc}', depth, writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        dct = super().asdict()
        for k in ('expected contribs contingency_table ' +
                  'adjusted_residuals pearson_residuals').split():
            del dct[k]
        self._write_dict(dct, depth, writer)
        self._write_line('expected:', depth, writer)
        self._write_dataframe(self.expected, depth + 1, writer)
        self._write_line('contributions:', depth, writer)
        self._write_dataframe(self.contribs, depth + 1, writer)
        self._write_line('pearson_residuals:', depth, writer)
        self._write_dataframe(self.pearson_residuals, depth + 1, writer)
        self._write_line('adjusted_residuals:', depth, writer)
        self._write_dataframe(self.adjusted_residuals, depth + 1, writer)
        self.write_associated(depth, writer)


class ChiSquareSignificanceTest(SignificanceTest):
    """A ChiSquare test using the 2x2 contigency table as input.

    """
    _NAME: ClassVar[str] = 'chisquare'

    def _compute_significance(self, data: SignificanceTestData) -> Evaluation:
        from scipy.stats import chi2_contingency
        dfc: pd.DataFrame = data.contingency_table
        chi2, p, dof, expected = chi2_contingency(dfc)
        dfe = pd.DataFrame(expected, columns=dfc.columns, index=dfc.index)
        return ChiSquareEvaluation(
            pvalue=p,
            alpha=data.alpha,
            statistic=chi2,
            dof=dof,
            expected=dfe,
            contingency_table=dfc)


SignificanceTestSuite._register_test(ChiSquareSignificanceTest)
