# Limitations and Future Work

## L1. Data Completeness and Literature Bias
Literature-derived datasets suffer from publication bias (over-reporting of positive results) and variations in reactor geometries, light sources, and experimental setups that are difficult to standardize.

## L2. Simplification of Structural Features
Crystal phase, surface facets, defect densities, and interface quality in heterojunctions are modeled as simple flags/descriptors. A more comprehensive representation of catalyst morphology (e.g., via graph neural networks on crystal structures) is a key area for future work.

## L3. Generalization to Unseen Classes
As demonstrated by the negative LOGO-CV performance when leaving out the major host class (TiO₂), machine learning models struggle to extrapolate to completely new chemical spaces. Active learning and transfer learning are recommended to mitigate this limitation.
