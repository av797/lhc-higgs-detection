# Missing Value Handling and Feature Preprocessing

## 1. Overview

The CERN ATLAS Higgs Boson dataset contains 30 input features describing different properties of particle collision events. Some of these features contain the value `-999`, which represents an undefined measurement rather than a genuine numerical value.

These values occur when a particular physical quantity cannot be calculated for an event, often because the required particles or jets are not present.

For example, the transverse momentum of a subleading jet cannot be calculated if an event contains fewer than two jets.

Rather than removing these events, all collision events will be retained through an imputation strategy.

## 2. Missing Value Imputation

Median imputation will be used to replace `-999` values in the affected features.

The median is the middle value when valid numerical measurements are arranged in ascending order. Values equal to `-999` are excluded when calculating the median.

Each affected feature will have its own median, calculated using the training dataset only.

For example, if the median of `PRI_jet_subleading_pt` is 40 GeV, every `-999` value in that feature will be replaced with 40 GeV.

The calculated median remains fixed throughout preprocessing. It is not recalculated for individual events or separately for the validation and test datasets.

Median imputation is selected because it provides a simple way to retain events while being less sensitive to extreme numerical values than mean imputation.

## 3. Missing Value Indicators

Replacing undefined measurements with median values alone would remove information about whether the original measurement was available.

To preserve this information, an additional binary indicator will be created for each feature containing `-999` values.

Each indicator will contain:

- `0`: The original measurement was available.
- `1`: The original measurement was undefined (`-999`).

For example, `PRI_jet_subleading_pt_missing` will indicate whether the original subleading jet momentum measurement was available.

These indicators allow the neural network to distinguish between an actual measured value and an imputed value.

There are 11 affected input features, so 11 additional binary indicators will be created.

## 4. Feature Standardisation

After imputation, the 30 numerical input features will be standardised using the training dataset's mean and standard deviation.

Standardisation transforms numerical values so that features have comparable scales.

The following formula will be applied:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) is the original feature value.
- \(\mu\) is the training-set mean.
- \(\sigma\) is the training-set standard deviation.
- \(z\) is the standardised value.

The same training-set parameters will be applied to validation and test data.

The 11 binary missing-value indicators will remain as 0 and 1 rather than being standardised.

## 5. Final Input Representation

The preprocessing strategy produces the following neural network input:

| Component | Number of features |
|---|---:|
| Original numerical features | 30 |
| Missing-value indicators | 11 |
| **Total input features** | **41** |

The resulting data will be converted into PyTorch tensors for use during neural network training.

## 6. Preventing Data Leakage

To ensure a fair evaluation, all preprocessing parameters will be calculated using training data only.

This includes:

- Median values used for imputation.
- Means used for standardisation.
- Standard deviations used for standardisation.

The same parameters will then be applied to the validation and test datasets without recalculating them.

This prevents information from the validation and test datasets from influencing the preprocessing of the training data.

## 7. Summary

The preprocessing strategy retains all collision events while addressing undefined physical measurements. Median imputation provides usable numerical values, and binary indicators preserve information about the original missing measurements.

Standardisation ensures that numerical features are on comparable scales before being passed into the PyTorch neural network.

The final model will therefore receive 41 input features: 30 standardised numerical features and 11 binary missing-value indicators.