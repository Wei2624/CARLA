from typing import List

import numpy as np
import pandas as pd

from carla.models.api import MLModel


def constraint_violation(
    mlmodel: MLModel, counterfactuals: pd.DataFrame, factuals: pd.DataFrame
) -> List[List[float]]:
    """
    Counts constraint violation per counterfactual

    Parameters
    ----------
    mlmodel: Black-box-model we want to discover
    counterfactuals: Normalized and encoded counterfactual examples
    factuals: Normalized and encoded factuals

    Returns
    -------

    """
    immutables = mlmodel.data.immutables

    df_decoded_cfs = mlmodel.data.inverse_transform(counterfactuals.copy())
    # Decode counterfactuals to compare immutables with not encoded factuals
    # df_decoded_cfs = counterfactuals.copy()
    # df_decoded_cfs = decode(
    #     mlmodel.data.encoder, mlmodel.data.categorical, df_decoded_cfs
    # )
    # df_decoded_cfs[mlmodel.data.continuous] = mlmodel.data.scaler.inverse_transform(
    #     df_decoded_cfs[mlmodel.data.continuous]
    # )
    # df_decoded_cfs[mlmodel.data.continuous] = df_decoded_cfs[
    #     mlmodel.data.continuous
    # ].astype(
    #     "int64"
    # )  # avoid precision error

    df_decoded_cfs[mlmodel.data.continuous] = df_decoded_cfs[
        mlmodel.data.continuous
    ].astype("int64")
    df_decoded_cfs = df_decoded_cfs[immutables]
    df_factuals = mlmodel.data.inverse_transform(factuals)[immutables]
    # df_factuals = mlmodel.data.inverse_transform(factuals)[immutables]

    logical = df_factuals != df_decoded_cfs
    logical = np.sum(logical.values, axis=1).reshape((-1, 1))

    return logical.tolist()
