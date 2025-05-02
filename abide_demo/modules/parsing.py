import argparse

__all__ = ["parse_args"]


def parse_args():
    """Parse command line arguments for the ABIDE multi-site autism classification demo.

    Returns
    -------
    parsed_args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.prog = "abide-site-adaptation-demo"
    parser.description = (
        "Demo on the use of domain adaptation to reduce site-dependency "
        "in functional connectivity features for autism classification "
        "using ABIDE dataset. Based on (Kunda et al., 2022) published in IEEE TMI 2022."
    )
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    pg = parser.add_argument_group("paths")

    pg.add_argument(
        "--input-dir",
        type=str,
        help="ABIDE dataset directory (required).",
        required=True,
    )

    pg.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to save the results (required).",
        required=True,
    )

    pg = parser.add_argument_group("preprocessing")

    pg.add_argument(
        "--atlas",
        type=str,
        default="cc200",
        choices=["aal", "cc200", "cc400", "dosenbach160", "ez", "ho", "tt"],
        help="ROIs to use.",
    )

    pg.add_argument(
        "--band-pass-filtering",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Apply band-pass filtering between 0.01Hz and 0.1Hz to the signals.",
    )

    pg.add_argument(
        "--global-signal-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply global signal regression on the signals.",
    )

    pg.add_argument(
        "--quality-checked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use quality checked data.",
    )

    pg = parser.add_argument_group("model")

    pg.add_argument(
        "--classifier",
        type=str,
        default="logistic",
        choices=["ridge", "svm", "logistic"],
        help="Classifier to use.",
    )

    pg.add_argument(
        "--mida",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use MIDA to reduce site-dependency.",
    )

    pg.add_argument(
        "--feature-extraction",
        nargs="+",
        type=str,
        default=["pearson"],
        choices=["covariance", "pearson", "partial", "tangent", "precision"],
        help="Sequence of feature extraction methods to use.",
    )

    pg = parser.add_argument_group("training")

    pg.add_argument(
        "--search-strategy",
        type=str,
        default="random",
        choices=["grid", "random"],
        help="Search strategy for hyperparameter tuning.",
    )

    pg.add_argument(
        "--num-search-iterations",
        type=int,
        default=10,
        help=(
            "Number of iterations for random search. "
            "Only used if search strategy is 'random'."
        ),
    )

    pg.add_argument(
        "--num-solver-iterations",
        type=int,
        default=100,
        help=(
            "Number of iterations for solver. "
            "Used for 'svm' or 'logistic' classifiers."
        ),
    )

    pg = parser.add_argument_group("evaluation")

    pg.add_argument(
        "--scoring",
        type=str,
        nargs="+",
        default=["accuracy"],
        choices=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "matthews_corrcoef",
        ],
        help=(
            "Scoring metrics to evaluate the model performance. "
            "The first metric will be used for hyperparameter tuning."
        ),
    )

    pg.add_argument(
        "--split",
        type=str,
        default="skf",
        choices=["skf", "lpgo"],
        help=(
            "Cross-validation strategy to use. "
            "'skf' for stratified k-fold, "
            "'lpgo' for leave-p-group-out cross-validation."
        ),
    )

    pg.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help=(
            "Number of folds for cross-validation. "
            "For 'lpgo', specifies number of test groups."
        ),
    )

    pg.add_argument(
        "--num-cv-repeats",
        type=int,
        default=1,
        help=(
            "Number of random repetition for cross-validation. "
            "Only used if split is 'skf'."
        ),
    )

    pg = parser.add_argument_group("runtime")

    pg.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel for hyperparameter tuning.",
    )

    pg.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random state for reproducibility.",
    )

    pg.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Set verbosity level for output during runtime.",
    )

    return parser.parse_args()
