from typing import Any, Dict

import numpy as np
import pandas as pd

from carla.models.api import MLModel
from carla.models.pipelining import encode, scale
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.face.library import graph_search
from carla.recourse_methods.processing import encode_feature_names


class Face(RecourseMethod):
    def __init__(self, mlmodel: MLModel, hyperparams: Dict[str, Any]) -> None:
        """
        Constructor for FACE method

        Restrictions
        ------------
        - Categorical features have to be binary
        - One-hot-encoding of binary feature should contain only one column per feature.

        Parameters
        ----------
        mlmodel : models.api.MLModel
            ML model to build counterfactuals for.
        hyperparams : dict
            Hyperparameter which are needed for FACE to generate counterfactuals.
            Structure:
            {
                "mode": str ['knn', 'epsilon'],
                "fraction": float [0 < x < 1]}  determines fraction of data set to be used to
                                                construct neighbourhood graph
        """
        super().__init__(mlmodel)
        self.mode = hyperparams["mode"]
        self.fraction = hyperparams["fraction"]

        # Normalize and encode data
        self._df_enc_norm = scale(
            self._mlmodel.scaler, self._mlmodel.data.continous, self._mlmodel.data.raw
        )
        self._df_enc_norm = encode(
            self._mlmodel.encoder, self._mlmodel.data.categoricals, self._df_enc_norm
        )
        self._df_enc_norm = self._df_enc_norm[self._mlmodel.feature_input_order]

        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )

    @property
    def fraction(self) -> float:
        return self._fraction

    @fraction.setter
    def fraction(self, x: float) -> None:
        if 0 < x < 1:
            self._fraction = x
        else:
            raise ValueError("Fraction has to be between 0 and 1")

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        if mode in ["knn", "epsilon"]:
            self._mode = mode
        else:
            raise ValueError("Mode has to be either knn or epsilon")

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Normalize and encode factual
        df_enc_norm_fact = scale(
            self._mlmodel.scaler, self._mlmodel.data.continous, factuals
        )
        df_enc_norm_fact = encode(
            self._mlmodel.encoder, self._mlmodel.data.categoricals, df_enc_norm_fact
        )
        df_enc_norm_fact = df_enc_norm_fact[self._mlmodel.feature_input_order]

        # >drop< factuals from dataset to prevent duplicates,
        # >reorder< and >add< factuals to top; necessary in order to use the index
        df_enc_norm_data = self._df_enc_norm.copy()
        cond = df_enc_norm_data.isin(df_enc_norm_fact).values
        df_enc_norm_data = df_enc_norm_data.drop(df_enc_norm_data[cond].index)
        df_enc_norm_data = pd.concat(
            [df_enc_norm_fact, df_enc_norm_data], ignore_index=True
        )

        list_cfs = []
        for i in range(df_enc_norm_fact.shape[0]):
            cf = graph_search(
                df_enc_norm_data,
                i,
                self._immutables,
                self._mlmodel,
                mode=self._mode,
                frac=self._fraction,
            )
            list_cfs.append(cf)

        df_cfs = pd.DataFrame(
            np.array(list_cfs), columns=self._mlmodel.feature_input_order
        )
        df_cfs[self._mlmodel.data.target] = np.argmax(
            self._mlmodel.predict_proba(df_cfs), axis=1
        )
        # Change all wrong counterfactuals to nan
        df_cfs.loc[df_cfs[self._mlmodel.data.target] == 0, :] = np.nan

        return df_cfs