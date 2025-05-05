import logging

import numpy as np
from kale.pipeline.mida_trainer import MIDATrainer
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, check_cv
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    StrOptions,
    validate_params,
)

__all__ = ["create_trainer"]

# Inverse regularization coefficients for the classifiers
# For Ridge (alpha) and MIDA (mu and eta), we use 1 / (2C)
C = np.logspace(start=-5, stop=15, num=20 + 1, base=2)

CLASSIFIER = {
    "logistic": LogisticRegression(),
    "svm": LinearSVC(),
    "ridge": RidgeClassifier(),
}

CLASSIFIER_GRID = {
    "logistic": {"C": C},
    "svm": {"C": C},
    "ridge": {"alpha": 1 / (2 * C)},
}

MIDA_GRID = {
    "num_components": [32, 64, 128, 256, None],
    "kernel": ["linear", "rbf"],
    "mu": 1 / (2 * C),
    "eta": 1 / (2 * C),
    "ignore_y": [True, False],
    "augment": [True, False],
}
MIDA_GRID = {f"domain_adapter__{key}": value for key, value in MIDA_GRID.items()}


@validate_params(
    {
        "classifier": [StrOptions({"logistic", "svm", "ridge"})],
        "mida": ["boolean"],
        "search_strategy": [StrOptions({"grid", "random"})],
        "cv": ["cv_object"],
        "scoring": [StrOptions(set(get_scorer_names())), list, None],
        "num_solver_iterations": [Interval(Integral, 1, None, closed="left")],
        "num_search_iterations": [Interval(Integral, 1, None, closed="left")],
        "num_jobs": [Integral, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    },
    prefer_skip_nested_validation=False,
)
def create_trainer(
    classifier="logistic",
    mida=False,
    search_strategy="grid",
    cv=None,
    scoring=None,
    num_solver_iterations=100,
    num_search_iterations=10,
    num_jobs=None,
    random_state=None,
    verbose=0,
):
    """Create a trainer for a classification model.

    Parameters
    ----------
    classifier : str, default="logistic"
        The classifier to use. Can be "logistic", "svm", or "ridge".

    mida : bool, default=False
        Whether to use MIDA for site-dependency reduction.

    search_strategy : str, default="grid"
        The search strategy for hyperparameter tuning. Can be "grid" or "random".

    cv : int, cross-validation generator, or iterable, default=None
        The cross-validation splitting strategy. If None, the default 5-fold
        cross-validation is used.

    scoring : str, list of str, callable, or None, default=None
        A single string or a list of strings to use as the scoring metric(s).
        If None, the default scoring metric for the classifier is used.

    num_solver_iterations : int, default=100
        The number of iterations for the solver. This is used to set the
        max_iter parameter of the classifier.

    num_search_iterations : int, default=10
        The number of iterations for the random search. This is only used
        if search_strategy is "random".

    num_jobs : int, default=None
        The number of jobs to run in parallel with joblib.Parallel. If None,
        the number of jobs is set to run on a single core.

    random_state : int, RandomState instance, or None, default=None
        The random seed for the random number generator. If None, the
        random state is not set.

    Returns
    -------
    trainer : sklearn.model_selection.BaseSearchCV or MIDATrainer
        The model trainer object. This can be either a GridSearchCV,
        RandomizedSearchCV, or MIDATrainer object.
    """
    if verbose > 0:
        logger = logging.getLogger("modeling.create_trainer")
        logger.setLevel(logging.INFO)

        logger.info(f"Creating trainer with classifier: {classifier}")
        logger.info(f"Using MIDA: {mida}")
        logger.info(f"Search strategy: {search_strategy}")
        logger.info(f"Scoring: {scoring}")
        logger.info(f"Number of solver iterations: {num_solver_iterations}")
        logger.info(f"Number of search iterations: {num_search_iterations}")
        logger.info(f"Number of jobs: {num_jobs}")
        logger.info(f"Random state: {random_state}")

    # Generate classifier with its parameter grid
    clf = clone(CLASSIFIER[classifier])
    clf.set_params(max_iter=num_solver_iterations, random_state=random_state)
    param_grid = clone(CLASSIFIER_GRID[classifier], safe=False)

    # Update with MIDA's parameters if we are using MIDA
    if mida:
        param_grid.update(MIDA_GRID)

    # Construct trainer
    trainer_args = {
        "cv": check_cv(cv, [0, 1], classifier=True),
        "scoring": scoring,
        "refit": scoring[0] if isinstance(scoring, list) else scoring,
        "n_jobs": num_jobs,
        "error_score": "raise",
        "verbose": verbose,
    }

    if verbose > 0:
        logger.info("Finished constructing trainer.")

    if mida:
        return MIDATrainer(
            estimator=clf,
            param_grid=param_grid,
            search_strategy=search_strategy,
            num_iter=num_search_iterations,
            random_state=random_state,
            **trainer_args,
        )

    if search_strategy == "grid":
        return GridSearchCV(estimator=clf, param_grid=param_grid, **trainer_args)

    return RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grid,
        n_iter=num_search_iterations,
        random_state=random_state,
        **trainer_args,
    )
