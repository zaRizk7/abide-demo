import logging
import os
import sys

import pandas as pd
from nilearn.datasets import fetch_abide_pcp
from sklearn.utils import check_random_state

from modules import (
    create_splitter,
    create_trainer,
    extract_functional_connectivity,
    parse_args,
    process_phenotypic_data,
)

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    if args.verbose > 0:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="[%(asctime)s] {%(filename)s:%(funcName)s:%(lineno)d} %(levelname)s - %(message)s",
        )

    # Prepare dataset
    atlas = f"rois_{args.atlas}"
    dataset = fetch_abide_pcp(
        data_dir=args.input_dir,
        quality_checked=args.quality_checked,
        derivatives=[atlas],
        band_pass_filtering=args.band_pass_filtering,
        global_signal_regression=args.global_signal_regression,
        verbose=args.verbose,
    )

    # Process phenotypic data
    phenotypes = process_phenotypic_data(dataset.phenotypic, args.verbose)

    # Extract functional connectivity features, labels, and groups/sites
    x = extract_functional_connectivity(
        dataset.get(atlas), args.feature_extraction, args.verbose
    )
    y = phenotypes["DX_GROUP"].map({"CONTROL": 0, "ASD": 1}).to_numpy()
    groups = phenotypes["SITE_ID"].to_numpy()

    # Drop unnecessary columns and encode categorical variables to one-hot variables
    # to produce the factor matrix which influences the data distribution.
    factors = phenotypes.drop(columns=["DX_GROUP"])
    factors = pd.get_dummies(factors, drop_first=True).to_numpy()

    # Create input for the trainer
    fit_args = {"x" if args.mida else "X": x, "y": y, "groups": groups}
    if args.mida:
        fit_args["factors"] = factors

    # Initialize random state
    random_state = check_random_state(args.random_state)

    # Create cv splitter and trainer
    cv = create_splitter(
        split=args.split,
        num_folds=args.num_folds,
        num_cv_repeats=args.num_cv_repeats,
        random_state=random_state,
        verbose=args.verbose,
    )

    # Initialize trainer and fit the trainer
    trainer = create_trainer(
        classifier=args.classifier,
        mida=args.mida,
        search_strategy=args.search_strategy,
        cv=cv,
        scoring=args.scoring,
        num_solver_iterations=args.num_solver_iterations,
        num_search_iterations=args.num_search_iterations,
        num_jobs=args.num_jobs,
        random_state=random_state,
        verbose=args.verbose,
    )

    trainer.fit(**fit_args)

    pd.DataFrame(trainer.cv_results_).to_csv(
        os.path.join(args.output_dir, "cv_results.csv"), index=False
    )

    if args.verbose > 0:
        logging.info("Finished training.")
