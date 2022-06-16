from ..api import Data
from typing import Callable, List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator

from carla.data.pipelining import (
    decode,
    descale,
    encode,
    fit_encoder,
    fit_scaler,
    scale,
)

# Custom data set implementations need to inherit from the Data interface
class mnist_bin(Data):
    def __init__(self):
        # The data set can e.g. be loaded in the constructor
        continuous = []
        for i in range(784):
            continuous.append('pixel' + str(i))
        self._continuous = continuous

        self.name = 'mnist_bin'
        df_train = pd.read_csv('train_mnist.csv')
        df_test = pd.read_csv('test_mnist.csv')
        # df =  df_train.append(df_test, ignore_index=True)

        self._df = df_train
        self._df_train = df_train
        self._df_test = df_test

        self._identity_encoding = True

        self.scaler: BaseEstimator = fit_scaler(
            "Identity", self.df[self.continuous]
        )
        self.encoder: BaseEstimator = fit_encoder(
            "OneHot_drop_binary", self.df[self.categorical]
        )



    # List of all categorical features
    @property
    def categorical(self):
        return []

    # List of all continuous features
    @property
    def continuous(self):
        return self._continuous

    # List of all immutable features which
    # should not be changed by the recourse method
    @property
    def immutables(self):
        return []

    # Feature name of the target column
    @property
    def target(self):
        return "label"

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test.copy()

    @property
    def scaler(self) -> BaseEstimator:
        """
        Contains a fitted sklearn scaler.

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        return self._scaler

    @scaler.setter
    def scaler(self, scaler: BaseEstimator):
        """
        Sets a new fitted sklearn scaler.

        Parameters
        ----------
        scaler : sklearn.preprocessing.Scaler
            Fitted scaler for ML model.

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        self._scaler = scaler

    @property
    def encoder(self) -> BaseEstimator:
        """
        Contains a fitted sklearn encoder:

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: BaseEstimator):
        """
        Sets a new fitted sklearn encoder.

        Parameters
        ----------
        encoder: sklearn.preprocessing.Encoder
            Fitted encoder for ML model.
        """
        self._encoder = encoder

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input for prediction into correct form.
        Only possible for DataFrames without preprocessing steps.

        Recommended to use to keep correct encodings and normalization

        Parameters
        ----------
        df : pd.DataFrame
            Contains raw (unnormalized and not encoded) data.

        Returns
        -------
        output : pd.DataFrame
            Prediction input normalized and encoded

        """
        output = df.copy()

        for trans_name, trans_function in self._pipeline:
            if trans_name == "encoder" and self._identity_encoding:
                continue
            else:
                output = trans_function(output)

        return output

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms output after prediction back into original form.
        Only possible for DataFrames with preprocessing steps.

        Parameters
        ----------
        df : pd.DataFrame
            Contains normalized and encoded data.

        Returns
        -------
        output : pd.DataFrame
            Prediction output denormalized and decoded

        """
        output = df.copy()

        for trans_name, trans_function in self._inverse_pipeline:
            output = trans_function(output)

        return output

    def get_pipeline_element(self, key: str) -> Callable:
        """
        Returns a specific element of the transformation pipeline.

        Parameters
        ----------
        key : str
            Element of the pipeline we want to return

        Returns
        -------
        Pipeline element
        """
        key_idx = list(zip(*self._pipeline))[0].index(key)  # find key in pipeline
        return self._pipeline[key_idx][1]

    def __init_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("scaler", lambda x: scale(self.scaler, self.continuous, x)),
            ("encoder", lambda x: encode(self.encoder, self.categorical, x)),
        ]

    def __init_inverse_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("encoder", lambda x: decode(self.encoder, self.categorical, x)),
            ("scaler", lambda x: descale(self.scaler, self.continuous, x)),
        ]

