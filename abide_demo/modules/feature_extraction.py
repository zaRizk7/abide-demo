import logging

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    StrOptions,
    validate_params,
)

__all__ = ["process_phenotypic_data", "extract_functional_connectivity"]

SELECTED_PHENOTYPES = [
    "SUB_ID",
    "SITE_ID",
    "SEX",
    "AGE_AT_SCAN",
    "FIQ",
    "HANDEDNESS_CATEGORY",
    "EYE_STATUS_AT_SCAN",
    "DX_GROUP",
]

MAPPING = {
    "SEX": {1: "MALE", 2: "FEMALE"},
    "HANDEDNESS_CATEGORY": {
        "L": "LEFT",
        "R": "RIGHT",
        "Mixed": "AMBIDEXTROUS",
        "Ambi": "AMBIDEXTROUS",
        "L->R": "AMBIDEXTROUS",
        "R->L": "AMBIDEXTROUS",
        "-9999": "LEFT",
        np.nan: "LEFT",
    },
    "EYE_STATUS_AT_SCAN": {1: "OPEN", 2: "CLOSED"},
    "DX_GROUP": {1: "CONTROL", 2: "ASD"},
}

AVAILABLE_FC_MEASURES = {
    "pearson": "correlation",
    "partial": "partial correlation",
    "tangent": "tangent",
    "covariance": "covariance",
    "precision": "precision",
}


@validate_params(
    {
        "data": [pd.DataFrame],
        "verbose": ["verbose"],
    },
    prefer_skip_nested_validation=False,
)
def process_phenotypic_data(data: pd.DataFrame, verbose=0):
    """Process phenotypic data to impute missing values and and encode handedness.

    Parameters
    ----------
    data : pd.DataFrame of shape (n_subjects, n_phenotypes)
        The phenotypes data to be processed.

    verbose : int, optional
        The verbosity level. The default is 0.
        verbose > 0 will log the current processing step.

    Returns
    -------
    processed_phenotypes : pd.DataFrame of shape (n_subjects, n_selected_phenotypes)
        The processed selected phenotype data with imputed values.
    """
    logger = logging.getLogger("feature_extraction.process_phenotypic_data")
    if verbose > 0:
        logger.setLevel(logging.INFO)
        logger.info("Imputing missing values and encoding handedness...")

    # Avoid in-place modification
    data = data.copy()

    # Check for missing values, either -9999 or NaN
    # and impute them with FIQ = 100 following original code.
    fiq = data["FIQ"].copy()
    data["FIQ"] = fiq.where((fiq != -9999) & (~np.isnan(fiq)), 100)

    # Encode categorical variables to be more explicit categorical
    # values. For handedness, if we found missing values, we
    # impute them by using 'LEFT' as default. Values
    # like 'Ambi', 'Mixed', 'L->R', and 'R->L' are mapped to
    # 'AMBIDEXTROUS'. The rest of the values are mapped to 'LEFT' or 'RIGHT'
    # for 'L' or 'R' respectively.
    for key in MAPPING:
        column = data[key].copy().map(MAPPING[key])
        data[key] = column.astype("category")

    if verbose > 0:
        logger.info("Imputation and encoding completed.")

    return data[SELECTED_PHENOTYPES].set_index("SUB_ID")


@validate_params(
    {
        "data": ["array-like"],
        "measures": [list],
        "verbose": ["verbose"],
    },
    prefer_skip_nested_validation=False,
)
def extract_functional_connectivity(data, measures=["pearson"], verbose=0):
    """Extract functional connectivity features from time series data.

    Parameters
    ----------
    data : list[array-like] of shape (n_subjects,)
        An array of numpy arrays, where each array is a time series of shape (t, n_rois).
        The time series data for each subject.

    measures : list[str], optional
        A list of connectivity measures to use for feature extraction.
        The default is ["pearson"].
        Supported measures are "pearson", "partial", "tangent", "covariance", and "precision".
        Multiple measures can be specified as a list to compose a higher-order measure.

    verbose : int, optional
        The verbosity level. The default is 0.
        verbose > 0 will log the current processing step.

    Returns
    -------
    features : array-like
        An array of shape (n_subjects, n_features) containing the extracted features.
        n_features is equal to `n_rois * (n_rois - 1) / 2` for each subjects.
    """
    if verbose > 0:
        logger = logging.getLogger("feature_extraction.extract_functional_connectivity")
        logger.setLevel(logging.INFO)
        logger.info("Extracting functional connectivity features...")
        logger.info(f"Using measures: {measures}")

    for i, k in enumerate(reversed(measures), 1):
        k = AVAILABLE_FC_MEASURES.get(k)

        # If it is the final transformation, vectorize and discard the diagonal
        final = i == len(measures)

        # We remove the diagonal given the pearson correlation will be equal to 1
        # and it is redundant. Then we vectorize the matrix to a vector
        # of shape (n_rois * (n_rois - 1) / 2, )
        measure = ConnectivityMeasure(kind=k, vectorize=final, discard_diagonal=final)
        data = measure.fit_transform(data)

    if verbose > 0:
        logger.info("Functional connectivity features extracted.")

    return data
