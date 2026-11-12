# Dataset Splitting Strategy

## Overview

The CERN ATLAS Higgs Boson Machine Learning dataset contains 818,238 collision events, divided into four original partitions: training (`t`), public leaderboard (`b`), private leaderboard (`v`) and unused (`u`).

The original partitions are retained when preparing the data to avoid mixing the competition's predefined datasets.

## Dataset Allocation

The following splitting strategy is used:

| Dataset | Original partition | Number of events | Purpose |
|---|---|---:|---|
| Training | `t` (80%) | 200,000 | Train the neural network |
| Validation | `t` (20%) | 50,000 | Monitor performance and tune hyperparameters |
| Testing | `b` + `v` | 550,000 | Evaluate final model performance |
| Excluded | `u` | 18,238 | Not used in the experiments |

The original training partition contains 250,000 events. This is divided into 80% training data and 20% validation data using a fixed random seed of 42. Stratified sampling is applied based on the target label to maintain approximately the same signal-to-background ratio in both subsets.

The original public and private leaderboard partitions are combined to form the final test dataset. These 550,000 events remain separate from the training and validation processes and are reserved for final evaluation.

The unused partition is excluded from the experiments.

## Justification

This strategy provides a clear separation between model training, validation and final testing while preserving the original dataset structure.

Using 200,000 training events provides a substantial number of examples for the PyTorch neural network to learn from. A separate validation set of 50,000 events allows model performance to be monitored during training, helping identify overfitting and guide hyperparameter selection.

The large test set of 550,000 events provides an extensive independent evaluation of the trained model. Keeping this data separate helps ensure that final performance metrics reflect the model's ability to classify previously unseen events.

A fixed random seed ensures that the training and validation split can be reproduced consistently.

## Reproducibility

The following settings are used when creating the training and validation subsets:

- Training proportion: 80%
- Validation proportion: 20%
- Random seed: 42
- Stratification: Target label (`Label`)
- Training framework: PyTorch

The physics event weight (`Weight`) is retained separately for investigating weighted training and evaluation. It is not included as an input feature to the neural network.

