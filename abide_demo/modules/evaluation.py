import logging

from sklearn.model_selection import LeavePGroupsOut, RepeatedStratifiedKFold
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    StrOptions,
    validate_params,
)

__all__ = ["create_splitter"]


@validate_params(
    {
        "split": [StrOptions({"skf", "lpgo"})],
        "n_splits": [Interval(Integral, 1, None, closed="left"), None],
        "n_repeats": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    },
    prefer_skip_nested_validation=False,
)
def create_splitter(
    split="skf",
    num_folds=5,
    num_cv_repeats=1,
    random_state=None,
    verbose=0,
):
    """
    Create a cross-validation splitter based on the specified strategy.

    Parameters
    ----------
    split : str, default="skf"
        The cross-validation strategy to use. Options are "skf" for stratified k-fold
        and "lpgo" for leave-p-group-out cross-validation.

    num_folds : int, default=5
        The number of folds for cross-validation. For "lpgo", it specifies the
        number of test groups.

    num_cv_repeats : int, default=1
        The number of repeats for cross-validation. Only used if split is "skf".

    random_state : int, default=None
        The random state for reproducibility. If None, the random state is not set.
        This is used for the stratified k-fold cross-validation.
        For "lpgo", this is not used.

    verbose : int, default=0
        The verbosity level. The default is 0.
        verbose > 0 will log the current processing step.

    Returns
    -------
    checked_cv : a cross-validator instance
        The cross-validation splitter object. This can be a RepeatedStratifiedKFold
        object for stratified k-fold cross-validation or a LeavePGroupsOut object
        for leave-p-group-out cross-validation.
    """
    if verbose > 0:
        logger = logging.getLogger("evaluation.create_splitter")
        logger.setLevel(logging.INFO)
        logger.info(f"Creating {split} cross-validation splitter...")
        logger.info(f"Number of folds: {num_folds}")
        logger.info(f"Number of repeats: {num_cv_repeats}")
        logger.info(f"Random state: {random_state}")
        logger.info("Finished creating cross-validation splitter.")

    if split == "lpgo":
        return LeavePGroupsOut(num_folds)

    return RepeatedStratifiedKFold(
        n_splits=num_folds, n_repeats=num_cv_repeats, random_state=random_state
    )
