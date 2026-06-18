# Supplementary Information: Machine Learning Assisted Photocatalyst Discovery

## S1. Dataset Details
* **Source**: Literature-mined glycerol photoreforming dataset.
* **Total Sample Count**: 706 experiments.
* **Target Feature**: log_HER = ln(HER + 1) where HER is in µmol g⁻¹ h⁻¹.
* **Features Included**: Experimental conditions (pH, concentration, wavelength, light power), semiconductor properties (bandgap, density, crystal structure), cocatalyst descriptors (work function, electronegativity, atomic radius), and engineered descriptors (effective carrier masses, CB/VB NHE potentials, exciton proxy).

## S2. Conformal Prediction Details
* **Method**: Split Conformal Prediction using MAPIE.
* **Confidence Level**: 90% (Alpha = 0.10).
* **Calibration Set Size**: 20% of training split (115 samples).
* **Empirical Coverage achieved on Test Set**: 86.15%.

## S3. Applicability Domain Boundaries
* **k-NN Distance**: Euclidean distance in the scaled active feature space. Threshold = mean + 2*std of training distances.
* **Isolation Forest**: Contamination parameter = 0.04. Threshold = 5th percentile score of training set.
* **Mahalanobis Distance**: Threshold = mean + 2*std of training distances.
* **Leverage (Williams Plot)**: Warning leverage $h^* = 3p/n$ where $p = 78$ and $n = 576$.
