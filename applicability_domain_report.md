# Applicability Domain Report

This report documents the applicability domain (AD) checking system used to validate predictions on test sets and virtual screening candidates.

## 1. Description of the 4 Methods
To provide a multi-dimensional assessment of predictive reliability, we implemented four distinct AD checking methods in scaled feature space:
1. **k-NN Distance**: Computes the mean Euclidean distance to the 5 nearest training neighbors. Establishes a threshold of $\mu_{train} + 2\sigma_{train}$.
2. **Isolation Forest**: Fits an isolation forest on training data to compute anomaly scores. Identifies outliers based on the 5th percentile of training anomaly scores.
3. **Leverage (Williams Plot)**: Uses the Hat Matrix to calculate leverage values ($h$) for each sample. A warning leverage threshold is established at $h^* = 3p/n$, where $p$ is the number of features and $n$ is the training set size.
4. **Mahalanobis Distance**: Computes distance taking covariance structure into account. Establishes a threshold of $\mu_{mahal\_train} + 2\sigma_{mahal\_train}$.

## 2. Test Set Reliability Classification

Our holdout test set ($n=130$) was classified into three trust categories:
- **Reliable** (inside all 4 boundaries): **119 samples (91.5%)**
- **Moderate Confidence** (inside 2–3 boundaries): **8 samples (6.2%)**
- **Outside Domain** (inside <= 1 boundary): **3 samples (2.3%)**

This high percentage of reliable predictions validates that the stratified split keeps test points within the model's interpolation domain.

## 3. Williams Plot Analysis
The Williams plot (standardized residuals vs leverage) is saved as `data/results/figures/fig_williams_plot.png`.
- The warning leverage limit is $h^* = 0.697$.
- Samples with standardized residuals $> 2.0$ or $< -2.0$ are potential response outliers, while samples with leverage $> h^*$ are structurally distinct.
- Only a small fraction of test points exceed these bounds, indicating excellent model stability.
