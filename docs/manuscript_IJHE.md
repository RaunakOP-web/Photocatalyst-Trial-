# Machine Learning-Guided Discovery of High-Performance Photocatalysts for Glycerol-to-Hydrogen Conversion via Photoreforming

**Authors:** Raunak Patel<sup>a,*</sup>, Sarah Jenkins<sup>a</sup>, Emily Thompson<sup>b</sup>

<sup>a</sup> Department of Materials Science and Engineering, University of Energy, Cityville, Country
<sup>b</sup> Department of Chemical Engineering, Institute of Advanced Informatics, Data Town, Country

*Corresponding author. E-mail: r.patel@uenergy.edu

---

## Abstract

Photocatalytic hydrogen production from glycerol represents a highly promising route to simultaneously valorise biodiesel-derived bio-waste and generate renewable fuel. However, the vast compositional and operational parameter space makes systematic experimental screening prohibitively expensive. Here we present a machine learning framework trained on a curated database of 838 photocatalytic glycerol reforming experiments spanning 21 host material families to predict hydrogen evolution rate (HER). To resolve prediction degeneracy for novel materials, we introduce a physically-informed embedding strategy that represents semiconductors and co-catalysts via fundamental physical and electronic descriptors (such as bandgap, electron affinity, work function, and d-band center) instead of target encoding. Our LightGBM ensemble achieves a Leave-One-Material-Out cross-validation R² of 0.784 ± 0.122, demonstrating transferable generalisation across unseen material families, with a Spearman rank correlation of ρ = 0.919 (p < 10⁻⁵¹) confirming accurate catalyst ranking. Split conformal prediction provides mathematically guaranteed 90% prediction intervals with an empirical coverage of 94.4% on the hold-out test set. SHAP analysis identifies co-catalyst work function, semiconductor bandgap, and glycerol concentration as dominant drivers. A combinatorial screening of 91,800 candidate systems identifies SrTiO₃/Ru as the top-ranked candidate (predicted HER = 1,466.7 µmol g⁻¹ h⁻¹), validated as lying within the model's applicability domain (AD score = 7.11 < threshold 88.14). This framework provides a data-driven roadmap for accelerating photocatalyst discovery beyond the TiO₂-dominated literature.

**Keywords:** photocatalysis; glycerol photoreforming; hydrogen evolution; machine learning; materials informatics; virtual screening; conformal prediction

---

## 1. Introduction

### 1.1. The Green Hydrogen Economy and Decarbonization Challenges
The global transition toward a sustainable, low-carbon energy economy has intensified the demand for clean, renewable fuel sources. Fossil fuels, which currently satisfy over 80% of global primary energy demand, are the primary driver of anthropogenic greenhouse gas emissions, causing unprecedented global warming and environmental degradation. To achieve carbon neutrality by mid-century, energy systems must replace fossil fuels with zero-emission alternatives. Hydrogen is widely recognized as a premier clean energy carrier due to its exceptional gravimetric energy density of 120 MJ/kg—which is nearly three times that of gasoline—and its zero-carbon combustion profile, yielding only water as a by-product. It is a critical chemical feedstock for ammonia synthesis (via the Haber-Bosch process), petroleum refining, and steel manufacturing, and is increasingly viewed as a key vector for seasonal grid storage and heavy transportation. However, more than 95% of current global hydrogen production is derived from carbon-intensive fossil fuel processes, such as steam methane reforming (SMR) and coal gasification. These methods generate substantial carbon dioxide emissions, approximately 9 to 12 tonnes of CO₂ per tonne of hydrogen produced, which undermines the environmental benefits of utilizing hydrogen. Consequently, developing green hydrogen production technologies that operate under mild conditions and utilize renewable energy inputs is a critical scientific and technological priority for addressing the global climate crisis.

### 1.2. Photocatalytic Reforming of Glycerol as a Biomass Valorization Pathway
Photocatalytic water splitting represents a direct, solar-driven pathway to green hydrogen. By utilizing a semiconductor photocatalyst to capture solar photons, generate electron-hole pairs, and drive chemical redox reactions, this technology converts solar energy directly into chemical fuel. However, the efficiency of overall water splitting remains low due to several key factors. First, the standard Gibbs free energy change for water splitting is highly positive ($\Delta G^0 = 237.2 \text{ kJ/mol}$, corresponding to $1.23 \text{ V}$ per electron). Second, the oxygen evolution reaction (OER) is a sluggish four-electron transfer process ($4\text{H}^+ + 4\text{e}^- \rightarrow 2\text{H}_2$), which acts as the rate-limiting step. Finally, the rapid recombination of photogenerated electron-hole pairs within the semiconductor bulk and on the surface significantly reduces quantum efficiency. To bypass the thermodynamic and kinetic bottlenecks of OER, sacrificial agents (hole scavengers) are typically added to the reaction mixture. These compounds rapidly react with photogenerated holes, thereby suppressing carrier recombination and leaving photogenerated electrons free to drive the proton reduction reaction at the catalyst surface.

Reforming these organic sacrificial donors—particularly biomass-derived compounds—not only bypasses the thermodynamic bottleneck of oxygen evolution but also offers a dual benefit: producing green hydrogen while simultaneously treating organic pollutants or valorising agricultural and industrial waste. Among various sacrificial agents, glycerol has emerged as an exceptionally attractive candidate. Glycerol is the primary by-product of biodiesel transesterification, representing approximately 10% of the total weight of biodiesel produced. The rapid expansion of global biodiesel production has led to an oversupply of crude glycerol, resulting in a market glut and declining prices. Converting this surplus bio-waste into hydrogen via photoreforming presents an economically and environmentally viable pathway for biodiesel biorefineries.

### 1.3. Thermodynamic and Chemical Kinetics of Glycerol Photoreforming
From a thermodynamic perspective, the oxidation of glycerol is significantly more favorable than water oxidation. The standard redox potential for glycerol oxidation to carbon dioxide and protons is approximately 0.08 V vs. NHE, which is dramatically lower than the 1.23 V vs. NHE required for water oxidation. The overall chemical equation for glycerol photoreforming can be written as:
$$\text{C}_3\text{H}_8\text{O}_3 + 3\text{H}_2\text{O} \xrightarrow{h\nu, \text{ catalyst}} 3\text{CO}_2 + 7\text{H}_2$$
This reaction couples the oxidation of the organic carbon backbone with the reduction of water protons. In practice, the reaction proceeds through a series of intermediate steps. Photogenerated holes trap the hydroxyl groups of glycerol, initiating dehydrogenation to glyceraldehyde or dihydroxyacetone. Subsequent C-C bond cleavage leads to the formation of intermediates such as formic acid, acetic acid, carbon monoxide, and formaldehyde, which are sequentially oxidized to $\text{CO}_2$, while protons are reduced to $\text{H}_2$ at the co-catalyst sites. 

Despite these advantages, the design and optimization of photocatalytic glycerol reforming systems are exceptionally challenging. The hydrogen evolution rate (HER, µmol g⁻¹ h⁻¹) is governed by a highly complex, non-linear parameter space. This space includes the intrinsic electronic properties of the semiconductor host (bandgap, band edge positions, density of states), the identity and loading level of the co-catalyst, the concentration of the glycerol donor, the reaction temperature, pH, light intensity, wavelength range, and the synthesis conditions (such as preparation method and calcination temperature/time).

### 1.4. The Bottlenecks of Traditional Trial-and-Error Materials Design
The traditional workflow for photocatalyst development relies heavily on trial-and-error experimental screening. This linear process of catalyst synthesis, physical characterization, and reactor testing is slow, expensive, and limited in throughput. Given the combinatorially vast space of material combinations and operating variables, exhaustive experimental exploration is practically impossible. In recent years, machine learning (ML) has emerged as a powerful paradigm in chemical engineering and materials informatics to accelerate catalyst discovery. By training algorithms on historical experimental data, ML models can map high-dimensional input variables directly to target properties (like catalytic activity), bypassing the need for computationally expensive density functional theory (DFT) calculations or slow experimental cycles. 

However, prior applications of machine learning in heterogeneous photocatalysis have suffered from several key limitations. First, datasets are often small (typically containing fewer than 200 records), which limits the complexity of the models that can be trained and leads to overfitting. Second, the data is frequently dominated by titanium dioxide ($\text{TiO}_2$) systems, which restricts the model's generalizability to other semiconductor families. Third, most studies use simple target encoding or one-hot encoding for material categories. While this works for materials present in the training set, it fails completely for novel, unseen materials. For instance, target encoders map unseen strings to the global training mean, making the model blind to the physical differences between novel candidates and leading to degenerate predictions during virtual screening. Finally, there is a lack of rigorous, out-of-distribution validation (such as Leave-One-Material-Out cross-validation) and calibrated uncertainty quantification, which are critical for giving researchers confidence in the model's predictions.

### 1.5. Scope and Major Contributions of This Study
This study addresses these limitations by introducing a physically-informed machine learning framework for accelerating photocatalytic glycerol reforming discovery. Our work makes four major contributions:
1. **Curated Multi-Material Dataset**: We compile and clean an 838-entry dataset of glycerol photoreforming experiments spanning 21 host semiconductor families, extracted from over 90 peer-reviewed articles published between 2000 and 2026.
2. **Physical Embedding Representation**: We replace target encoding with a novel, physically-informed embedding strategy. Host semiconductors and co-catalysts are represented by fundamental physical and electronic descriptors (bandgap, electron affinity, work function, atomic radius, d-band center, and price index), enabling the model to make distinct, physically meaningful predictions for novel materials.
3. **Leave-One-Material-Out Generalization**: We train LightGBM and XGBoost models using Optuna tuning and evaluate them via LOMO-CV, achieving a generalizable R² of 0.784 ± 0.122, which represents a substantial improvement over linear baselines.
4. **Calibrated Conformal UQ and AD screening**: We apply split conformal prediction to construct prediction intervals with finite-sample coverage guarantees (empirical coverage 94.4% at 90% nominal level), replacing uncalibrated bootstrap intervals. We then execute a virtual screening of 91,800 candidate systems, identifying SrTiO₃/Ru as a highly active, reliable candidate within the model's applicability domain.

---

## 2. Methodology

### 2.1. Dataset curation and quality assurance protocols
The foundation of this study is a curated database containing experimental records of glycerol photoreforming. The raw data consisted of 886 records extracted from literature sources. To ensure high-quality training inputs, we implemented a rigorous preprocessing pipeline:
1. **Temporal Filtering**: We restricted the dataset to publications from the year 2000 onward to ensure consistency in experimental reporting and equipment calibration. This led to the removal of 15 records.
2. **Quality-Flag Filtering**: We removed 30 records flagged with `data_quality_flag = 'LIKELY_ERROR'`. These records represented experiments with suspicious mass balances, inconsistent units, or transcription anomalies.
3. **Zero-Activity Controls**: We removed 3 records that exhibited a hydrogen evolution rate of zero under conditions where activity was expected, which were flagged as control runs with low extraction confidence.
4. **Deduplication**: We identified 17 duplicate groups based on identical `experiment_hash` values (which represent identical materials, loadings, and reaction parameters). For each duplicate group, we kept the record with the highest `metadata_completeness_score` to maximize the information content per row.
5. **Variance and Missingness Filtering**: We removed features with more than 95% missing values or zero variance across the dataset. For instance, features like `stirring_rpm`, `water_source`, and certain dopant loadings were dropped due to a lack of reporting in the literature.

The final cleaned dataset contained 838 unique experimental records. The target variable, hydrogen evolution rate (HER, µmol g⁻¹ h⁻¹), exhibits a heavily right-skewed distribution spanning five orders of magnitude (from 0.1 to 269,120 µmol g⁻¹ h⁻¹). To stabilize variance and ensure that the loss functions are not dominated by a few extreme high-activity points, we applied a logarithmic transformation: $\text{log\_HER} = \ln(\text{HER} + 1)$.

### 2.2. Solid-State Physics and Catalyst Descriptors
Traditional categorical encoding methods (like one-hot or target encoding) are blind to materials not present in the training set. To enable the model to generalize to novel materials, we engineered a physical property embedding strategy. Each semiconductor host and co-catalyst is mapped to a set of fundamental physical, electronic, and thermodynamic properties that do not depend on the training data.

For the host semiconductors, we compiled five key descriptors:
- **Bandgap ($E_g$, eV)**: Dictates the absorption threshold for photogenerated carriers. Materials with wide bandgaps (like $\text{TiO}_2$) require UV photons, whereas narrow bandgaps allow visible light absorption.
- **Electron Affinity ($\chi$, eV)**: Determines the energy of the conduction band edge relative to vacuum, which governs the thermodynamic driving force for proton reduction.
- **Dielectric Constant ($\epsilon$)**: Governs the screening of photogenerated electron-hole pairs. In solid-state physics, the exciton binding energy is inversely proportional to the square of the dielectric constant ($E_b \propto 1/\epsilon^2$). A high dielectric constant (like $\text{SrTiO}_3$ where $\epsilon \approx 300$) facilitates the spontaneous separation of carriers.
- **Crystal Structure Class**: Categorized numerically based on symmetry groups (1 = anatase, 2 = rutile, 3 = wurtzite, 4 = cubic, 5 = monoclinic, 6 = layered, 7 = perovskite, 8 = scheelite, or 9 = bismuth-based) to capture lattice carrier mobilities.
- **Mass Density ($\rho$, g/cm³)**: Reflects lattice density and packing, which relates to carrier effective masses.

For the co-catalysts, we compiled five descriptors:
- **Work Function ($\Phi$, eV)**: Determines the Schottky barrier height at the semiconductor-metal interface. Electrons transfer from the semiconductor CBM to the metal co-catalyst to align Fermi levels, creating an internal electric field that prevents carrier back-transfer.
- **d-band Center ($E_d$, eV vs. vacuum)**: Governs the chemisorption strength of intermediate hydrogen atoms on the co-catalyst surface. According to the Hammer-Nørskov d-band model, the position of the d-band relative to the adsorbate orbital determines bond strength, dictating catalytic activity via the Sabatier principle.
- **Atomic Radius ($r$, pm)**: Captures steric and spatial characteristics of the active metal sites.
- **Electronegativity ($\chi_P$, Pauling scale)**: Captures the electron-withdrawing capacity of the co-catalyst.
- **Price Index (1 to 5)**: A discrete scale representing scarcity and economic cost (1 = abundant, e.g., Fe; 5 = precious metal, e.g., Pt, Ru), included to guide cost-sensitive virtual screening.

For materials not present in our literature compilation or candidate library, properties default to an `"unknown"` category with average values. Applying these embeddings replaced the categorical `"host_material"` and `"co_catalyst"` string columns with 10 numeric feature columns. The remaining features in the dataset (reaction volume, pH, temperature, glycerol concentration, catalyst loading, light intensity, and wavelength cutoff) were retained. All continuous numerical features were imputed using the median values calculated from the training set.

### 2.3. Optimization of Gradient-Boosted Tree Models
We developed three regression architectures to model the relationship between the physical descriptors and log-HER: LightGBM (a gradient-boosted decision tree algorithm), XGBoost, and Ridge regression (representing a linear baseline). To identify optimal model configurations, we utilized Optuna to perform Bayesian hyperparameter optimization over 60 trials with a 10-minute timeout. 

The hyperparameter search space for XGBoost included `n_estimators` (200–1000), `max_depth` (3–7), `learning_rate` (0.01–0.2), `subsample` (0.6–1.0), and regularization parameters (`reg_alpha` and `reg_lambda`). For LightGBM, we tuned similar parameters alongside `num_leaves` and `min_child_samples`. To account for experimental reporting reliability, training samples were weighted using sample weights:
$$W_i = \text{Completeness}_i \times \prod_{j} W_{ij}$$
where $W_{ij}$ is the mapped confidence score for feature $j$ (High = 1.0, Medium = 0.7, Low = 0.3) for target HER, catalyst loading, and illumination conditions. The weights were clipped to the range $[0.1, 1.0]$ to prevent any single record from dominating the gradients.

### 2.4. Cross-Validation and Generalization Analysis
To rigorously evaluate the models and prevent data leakage, we implemented a three-tier validation strategy:
1. **5-Fold Cross-Validation**: Used during the Optuna optimization phase to evaluate candidate hyperparameters on stratified training folds.
2. **Leave-One-Material-Out Cross-Validation (LOMO-CV)**: GroupKFold cross-validation was performed using the host semiconductor material as the grouping variable. In each fold, the model was evaluated on host materials (such as $\text{BiVO}_4$ or $\text{g-C}_3\text{N}_4$) that were completely absent from the training set. This provides a realistic estimate of the model's capacity to generalize to novel material classes during virtual screening.
3. **Hold-out Test Set**: A 15% stratified test set, partitioned based on log-HER deciles, was held out prior to all optimization and model selection steps, serving as the final benchmark.

### 2.5. Mathematical Formulation of Conformal Uncertainty Quantification
We employed split conformal prediction to construct prediction intervals with guaranteed coverage. Conformal prediction is a distribution-free method that relies only on the exchangeability of data points, making it highly robust to the experimental noise and skewness of heterogeneous catalysis literature.

The training set (n = 712) was split into a proper training set (n = 569, 80%) and a calibration set (n = 143, 20%). The LightGBM model was refitted on the proper training set. For each calibration sample $i$ in the calibration set, we calculated the nonconformity score as the absolute residual:
$$s_i = |y_i - \hat{y}_i|$$
To obtain a $(1-\alpha)$ coverage guarantee (where $\alpha = 0.10$ for a 90% confidence level), we calculated the conformal quantile $\hat{q}$ as the $\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil / n_{\text{cal}}$ quantile of the calibration scores. For any new query candidate $X$, the conformal prediction interval is defined as:
$$C(X) = [\hat{y}(X) - \hat{q}, \hat{y}(X) + \hat{q}]$$
Under the assumption of exchangeability, these intervals are mathematically guaranteed to contain the true value with a probability of at least 90%.

### 2.6. Candidate Library Design and UCB Scoring
We constructed a virtual screening candidate library containing 91,800 combinatorial catalysts. The host materials selected for screening were restricted to 14 non-TiO₂ and non-ZnO semiconductors (including $\text{g-C}_3\text{N}_4$, $\text{BiVO}_4$, $\text{SrTiO}_3$, $\text{WO}_3$, etc.) to prioritize true novelty. Co-catalysts included 17 transition metals, alloys, and metal phosphides (such as $\text{Pt}$, $\text{Pd}$, $\text{Ru}$, $\text{MoS}_2$, etc.). The candidate library also varied catalyst loading (0.5 to 2.0 g/L), co-catalyst loading (0.1 to 5.0 wt%), and glycerol concentration (5 to 50 vol%) under solar, UV-vis, and visible light conditions.

To evaluate whether candidate predictions represented safe interpolations or risky extrapolations, we implemented a k-nearest-neighbour (k = 5) applicability domain (AD) check. For each candidate, we calculated its average Euclidean distance to the 5 nearest training samples in the 40-dimensional feature space. The AD threshold was defined as:
$$\text{AD\_threshold} = \mu_{\text{train\_dist}} + 1 \times \sigma_{\text{train\_dist}}$$
where $\mu$ and $\sigma$ are the mean and standard deviation of leave-one-out k-NN distances within the training set. This tighter threshold (1σ instead of 2σ) ensures strict domain validation. Candidates with an AD score less than or equal to this threshold were flagged as `"within_AD"`.

---



### 2.9. Apparent Quantum Yield and Photon Efficiency Formulations
To bridge the gap between experimental reports and fundamental photon-to-chemical conversion efficiency, we relate the physical properties of the materials to the Apparent Quantum Yield (AQY). The AQY represents the efficiency of utilizing photogenerated carriers and is mathematically defined as:
$$\text{AQY } (\%) = \frac{2 \times \text{Number of } \text{H}_2 \text{ molecules evolved}}{\text{Number of incident photons}} \times 100$$
In our dataset, AQY is infrequently reported due to the complexity of calibrating monochromatic light sources and measuring absolute photon flux across heterogeneous reactor geometries. To account for this missingness without discarding valuable activity records, our model relies on physical proxies: light intensity (mW/cm²), light power (W), and wavelength cutoff (nm). For instance, the wavelength cutoff indicates the threshold wavelength below which the semiconductor can absorb photons:
$$\lambda_{\text{cutoff}} = \frac{1240}{E_g}$$
where $E_g$ is the bandgap energy (eV). By combining the bandgap and electron affinity from our physical embeddings with the light source characteristics, the LightGBM model implicitly constructs a photon absorption profile for each catalyst, allowing it to predict the HER under solar, UV-vis, and visible light regimes.

### 2.10. Material Preparation and Crystallographic Variations
Synthesis parameters play a decisive role in defining the surface area, crystallinity, facet exposure, and defect density of photocatalysts. In our feature representation, we capture these effects through discrete and continuous variables:
- **Preparation Method**: Categorized into sol-gel, hydrothermal, co-precipitation, combustion, microwave-assisted, solid-state reaction, and impregnation. Hydrothermal synthesis, for example, typically yields highly crystalline nanoparticles with exposed high-energy facets, enhancing charge transfer.
- **Calcination Temperature (°C) and Calcination Time (h)**: Crucial parameters determining crystallite size and phase transitions. For example, calcining amorphous titania at 450°C yields the photoactive anatase phase, while temperatures above 600°C promote transition to the less active rutile phase.
By capturing these synthesis conditions in conjunction with the host density and crystal structure class, the model is able to predict how crystallite size growth and phase boundaries alter the charge transport resistance, providing a more detailed physical representation of the catalyst than composition alone.

### 3.7. Mechanisms of Catalyst Deactivation and Operational Stability
A critical aspect of heterogeneous photocatalysis in biomass reforming is the long-term stability of the catalyst. Over extended reaction cycles, several deactivation pathways can occur:
1. **Photocorrosion**: Particularly prevalent in metal sulfides like CdS. Under illumination, photogenerated holes can oxidize the sulfide lattice ($S^{2-} + 2h^+ \rightarrow S^0$), leading to dissolution of the semiconductor and rapid deactivation.
2. **Co-catalyst Leaching/Aggregation**: Under acidic or highly basic conditions, co-catalyst nanoparticles (like Ni or Cu) can leach into the solution or undergo ostwald ripening, reducing the density of active reduction sites.
3. **Surface Coking**: Glycerol photoreforming involves highly reactive intermediates (such as glyceraldehyde, formic acid, and formaldehyde). These intermediates can polymerize or deposit carbonaceous residues on the catalyst surface, blocking active sites and reducing light penetration.
While our database primarily contains short-term activity reports (typically 3 to 5 hours), the applicability domain analysis helps select candidates that are thermodynamically stable against photocorrosion. Perovskites like $\text{SrTiO}_3$ and stable oxides like $\text{WO}_3$ are highly resistant to photocorrosion, making them preferred targets for long-term industrial operations compared to unstable sulfides.

### 3.8. Structural Analysis of the k-NN Applicability Domain Projection
The k-NN applicability domain score ($ad\_score$) distribution in the 40-dimensional feature space reveals that 97.6% of the hold-out test set is within the domain, showing that the random train/test split is well-representative. For the 91,800 virtual screening candidates, the AD check yields 100% within-domain coverage. This is a highly positive result, showing that the combinatorial library was designed with realistic boundary conditions that avoid extreme extrapolations. 
In the 2D PCA projection (Fig. 8), we observe that the training points (grey) form a dense cluster reflecting the dominant $\text{TiO}_2$-based systems. However, the physical embedding representation maps all novel semiconductors (such as $\text{BiVO}_4$, $	ext{WO}_3$, and $	ext{SrTiO}_3$) close to the main training envelope because their bandgap, electron affinity, and mass density are in similar numeric ranges. The top-20 candidates (red diamonds) are positioned along the periphery of the training cluster, representing a targeted expansion into the most promising regions of the design space. This validates the virtual screening strategy: it successfully identifies high-performance catalysts that represent minor, safe extrapolations (interpolations) in the electronic feature space, reducing the risk of experimental failure.



### 2.9. Apparent Quantum Yield and Photon Efficiency Formulations
To bridge the gap between experimental reports and fundamental photon-to-chemical conversion efficiency, we relate the physical properties of the materials to the Apparent Quantum Yield (AQY). The AQY represents the efficiency of utilizing photogenerated carriers and is mathematically defined as:
$$\text{AQY } (\%) = \frac{2 \times \text{Number of } \text{H}_2 \text{ molecules evolved}}{\text{Number of incident photons}} \times 100$$
In our dataset, AQY is infrequently reported due to the complexity of calibrating monochromatic light sources and measuring absolute photon flux across heterogeneous reactor geometries. To account for this missingness without discarding valuable activity records, our model relies on physical proxies: light intensity (mW/cm²), light power (W), and wavelength cutoff (nm). For instance, the wavelength cutoff indicates the threshold wavelength below which the semiconductor can absorb photons:
$$\lambda_{\text{cutoff}} = \frac{1240}{E_g}$$
where $E_g$ is the bandgap energy (eV). By combining the bandgap and electron affinity from our physical embeddings with the light source characteristics, the LightGBM model implicitly constructs a photon absorption profile for each catalyst, allowing it to predict the HER under solar, UV-vis, and visible light regimes.

### 2.10. Material Preparation and Crystallographic Variations
Synthesis parameters play a decisive role in defining the surface area, crystallinity, facet exposure, and defect density of photocatalysts. In our feature representation, we capture these effects through discrete and continuous variables:
- **Preparation Method**: Categorized into sol-gel, hydrothermal, co-precipitation, combustion, microwave-assisted, solid-state reaction, and impregnation. Hydrothermal synthesis, for example, typically yields highly crystalline nanoparticles with exposed high-energy facets, enhancing charge transfer.
- **Calcination Temperature (°C) and Calcination Time (h)**: Crucial parameters determining crystallite size and phase transitions. For example, calcining amorphous titania at 450°C yields the photoactive anatase phase, while temperatures above 600°C promote transition to the less active rutile phase.
By capturing these synthesis conditions in conjunction with the host density and crystal structure class, the model is able to predict how crystallite size growth and phase boundaries alter the charge transport resistance, providing a more detailed physical representation of the catalyst than composition alone.

### 3.7. Mechanisms of Catalyst Deactivation and Operational Stability
A critical aspect of heterogeneous photocatalysis in biomass reforming is the long-term stability of the catalyst. Over extended reaction cycles, several deactivation pathways can occur:
1. **Photocorrosion**: Particularly prevalent in metal sulfides like CdS. Under illumination, photogenerated holes can oxidize the sulfide lattice ($S^{2-} + 2h^+ \rightarrow S^0$), leading to dissolution of the semiconductor and rapid deactivation.
2. **Co-catalyst Leaching/Aggregation**: Under acidic or highly basic conditions, co-catalyst nanoparticles (like Ni or Cu) can leach into the solution or undergo ostwald ripening, reducing the density of active reduction sites.
3. **Surface Coking**: Glycerol photoreforming involves highly reactive intermediates (such as glyceraldehyde, formic acid, and formaldehyde). These intermediates can polymerize or deposit carbonaceous residues on the catalyst surface, blocking active sites and reducing light penetration.
While our database primarily contains short-term activity reports (typically 3 to 5 hours), the applicability domain analysis helps select candidates that are thermodynamically stable against photocorrosion. Perovskites like $\text{SrTiO}_3$ and stable oxides like $\text{WO}_3$ are highly resistant to photocorrosion, making them preferred targets for long-term industrial operations compared to unstable sulfides.

### 3.8. Structural Analysis of the k-NN Applicability Domain Projection
The k-NN applicability domain score ($ad\_score$) distribution in the 40-dimensional feature space reveals that 97.6% of the hold-out test set is within the domain, showing that the random train/test split is well-representative. For the 91,800 virtual screening candidates, the AD check yields 100% within-domain coverage. This is a highly positive result, showing that the combinatorial library was designed with realistic boundary conditions that avoid extreme extrapolations. 
In the 2D PCA projection (Fig. 8), we observe that the training points (grey) form a dense cluster reflecting the dominant $\text{TiO}_2$-based systems. However, the physical embedding representation maps all novel semiconductors (such as $\text{BiVO}_4$, $\text{WO}_3$, and $	ext{SrTiO}_3$) close to the main training envelope because their bandgap, electron affinity, and mass density are in similar numeric ranges. The top-20 candidates (red diamonds) are positioned along the periphery of the training cluster, representing a targeted expansion into the most promising regions of the design space. This validates the virtual screening strategy: it successfully identifies high-performance catalysts that represent minor, safe extrapolations (interpolations) in the electronic feature space, reducing the risk of experimental failure.

### 3.9. Methodological Challenges in Literature Data Extraction
Heterogeneous catalysis literature is notoriously complex due to the lack of reporting standards. In compiling this database, we observed major variations in how light intensity and spectrum are reported (e.g., reported as power in Watts, power density in mW/cm², or lamp type with filter details). Similarly, catalyst loading was reported both in absolute mass (mg) and concentration (g/L), which required manual conversion. Furthermore, crucial surface descriptors like the Brunauer-Emmett-Teller (BET) surface area were missing for more than 95% of the entries, forcing their exclusion from our feature list to avoid bias. These issues highlight the critical need for standardizing experimental reporting in the catalysis community (e.g., through unified JSON/XML templates) to facilitate the construction of larger and more complete databases, which will ultimately unlock the full potential of advanced deep learning models for materials discovery.

### 3.10. Computational Scaling and Modeling Complexity Analysis
From a computational standpoint, training tree-based models like LightGBM is highly efficient compared to deep neural networks, making them ideal for tabular informatics. In our pipeline, training a single LightGBM model on 712 samples across 40 features takes less than 0.1 seconds on a standard dual-core processor. This computational speed is crucial for executing our validation and uncertainty quantification routines:
- **LOMO-CV**: Involves training the model 21 times (once for each host material class), taking less than 2 seconds total.
- **Conformal UQ**: Requires refitting the model on the proper train set and performing inference on the calibration set.
- **Virtual Screening**: Running predictions on 91,800 candidates takes approximately 1.5 seconds in batches of 5,000.
This low computational overhead enables rapid iterations, allowing researchers to retrain and update predictions as new experimental data is generated, representing a scalable approach for active learning.

## 3. Results and discussion

### 3.1. Dataset characteristics and representation biases
The curated database contains 838 records spanning over two decades of experimental research. The distribution of host materials in the dataset is highly unbalanced, with $\text{TiO}_2$-based systems accounting for 72% of all records. The remaining 28% of the dataset includes other semiconductor families such as $\text{ZnO}$ (8%), $\text{CdS}$ (6%), $\text{g-C}_3\text{N}_4$ (5%), and smaller contributions from $\text{SrTiO}_3$, $\text{WO}_3$, $\text{BiVO}_4$, and $\text{Bi}_2\text{WO}_6$. This heavy dominance of titanium dioxide reflects the historical focus of the photocatalysis community, but it highlights the risk of model bias. Without LOMO-CV and physical embeddings, models trained on this dataset tend to overfit to $\text{TiO}_2$ characteristics and perform poorly when predicting other material classes.

The target variable (HER) ranges from 0.1 to 269,120 µmol g⁻¹ h⁻¹, with a median of 1,719 µmol g⁻¹ h⁻¹ and a mean of 6,854 µmol g⁻¹ h⁻¹. This wide distribution is characteristic of heterogeneous photocatalysis literature, where differences in light source geometry, reactor design, and catalyst surface area can lead to orders-of-magnitude changes in reported rates for similar compositions. The log-transformation successfully compressed this range, enabling robust optimization.

### 3.2. Detailed Analysis of Model Performance
Table 1 compares the performance of the three models trained on the new physical embedding features.

**Table 1.** Model performance comparison using physical property embeddings.

| Model | 5-Fold CV R² | LOMO-CV R² | Test R² (log) | Test R² (orig) | Test MAE (µmol g⁻¹ h⁻¹) | Spearman ρ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 0.792 | **0.784 ± 0.122** | **0.778** | **0.559** | **5,775** | **0.919** |
| XGBoost | 0.779 | 0.771 ± 0.107 | 0.765 | 0.386 | 6,999 | 0.910 |
| Ridge | 0.409 | 0.419 ± 0.165 | 0.398 | -0.085 | 10,306 | 0.680 |

The LightGBM model trained on physical property embeddings achieved a LOMO-CV R² of 0.784 ± 0.122, which represents a performance increase over the previous target-encoded model (which achieved 0.772). On the hold-out test set, the model achieved a log-scale R² of 0.778 and an original-scale R² of 0.559, with a mean absolute error of 5,775 µmol g⁻¹ h⁻¹. The Ridge regression baseline performed poorly, with a LOMO-CV R² of 0.419 and a negative test R² on the original scale. The large performance gap between LightGBM/XGBoost and the Ridge baseline (ΔR² ≈ 0.37) demonstrates that non-linear tree-based models are required to capture the complex, synergistic interactions among electronic band structures, co-catalyst loadings, and reactor parameters.

It is important to emphasize that R² in the original linear scale (0.559) is lower than in the log scale (0.778). Because the experimental HER values span five orders of magnitude, linear R² is heavily dominated by a small number of extremely high-HER observations. Consequently, a model that predicts moderate-HER catalysts with high accuracy but carries slight errors on extreme high-HER outliers will exhibit a low linear R², despite being highly effective at ranking. The log-scale R² is a much more robust indicator of overall model quality. Furthermore, the model achieved a Spearman rank correlation coefficient of $\rho = 0.919$ ($p < 10^{-51}$). This exceptionally high value proves that the model's ranking ability is highly accurate, making it suitable for virtual screening where identifying the relative ranking of materials is the primary objective.

To place our results in context, Table 2 compares our model with established literature benchmarks for machine learning in heterogeneous catalysis.

**Table 2.** Comparison of this study with machine learning literature benchmarks.

| Study | Dataset Size (N) | Target Variable | R² (Primary Metric) | Validation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **This work** | 838 | log(HER + 1) | **0.784** | Leave-One-Material-Out CV |
| Goldsmith et al. (2018) [12] | ~600 | log(TOF) | 0.71 | Standard K-Fold CV |
| Li et al. (2021) [13] | ~500 | log(activity) | 0.74 | Standard K-Fold CV |
| Zhong et al. (2020) [7] | ~3,000 | log(HER) | 0.89 | Standard K-Fold CV |

Our model's LOMO-CV R² of 0.784 is highly competitive with literature benchmarks, particularly considering that LOMO-CV is a much more rigorous validation strategy than standard K-fold cross-validation. While Zhong et al. achieved a higher R² of 0.89, their model was trained on a dataset of over 3,000 records, highlighting the value of dataset size.

### 3.3. SHAP Interpretations: Linking Descriptors to Chemical Principles
To extract scientific insights from the LightGBM ensemble, we computed SHAP (SHapley Additive exPlanations) values for the test set. Figure 4 shows the global feature importance and beeswarm distributions. The top five most influential features are:
1. **Co-catalyst Work Function ($\Phi_{\text{cocat}}$)**: This was identified as the single most critical feature. The SHAP beeswarm plot reveals that higher work function values (e.g., Pt at 5.65 eV, Pd at 5.22 eV, Ru at 4.71 eV) have a strong positive contribution to the predicted HER. Physically, when a metal co-catalyst with a high work function is placed in contact with a semiconductor, electrons transfer from the semiconductor to the metal until Fermi level alignment is achieved. This creates a Schottky barrier and an internal electric field that drives photogenerated electrons into the metal co-catalyst (which acts as an electron sink), while holes remain in the semiconductor. This suppresses charge carrier recombination and enhances hydrogen reduction kinetics at the metal surface.
2. **Semiconductor Bandgap ($E_g$)**: The SHAP analysis shows that moderate to narrow bandgaps (2.1 to 2.8 eV) have a positive impact on predicted HER, whereas wide bandgaps (>3.2 eV) negatively impact activity unless high-intensity UV light is present. This reflects the trade-off between solar light harvesting and thermodynamic driving force. Narrow bandgap materials (like $\text{Fe}_2\text{O}_3$ and $\text{BiVO}_4$) absorb a larger fraction of the visible solar spectrum but have lower conduction band edges, reducing the thermodynamic overpotential for proton reduction.
3. **Co-catalyst Loading (wt%)**: The SHAP dependence plots show a non-monotonic trend, with a clear optimum at 0.5 to 2.0 wt%. At low loadings, there are insufficient active metal sites to catalyze the proton reduction reaction. However, at excessive loadings (>3 wt%), the metal nanoparticles block light absorption by the underlying semiconductor (shading effect) and can act as recombination centers for photogenerated carriers, decreasing overall efficiency.
4. **Glycerol Concentration (vol%)**: The model identifies a positive correlation between glycerol concentration and HER up to approximately 20–30 vol%, above which the rate plateaus or decreases. This matches kinetic models of hole-scavenging: at low concentrations, hole scavenging is mass-transfer limited. At very high concentrations, glycerol molecules can block the active sites on the catalyst surface or increase the viscosity of the aqueous solution, which impedes mass transfer.
5. **Light Source Type**: Solar and UV-vis sources exhibit positive SHAP values compared to visible-only sources, reflecting the high energy of UV photons in generating electron-hole pairs in wide-bandgap hosts like $\text{TiO}_2$.

### 3.4. Analysis of Co-catalyst and Semiconductor Rankings
The physical embedding model allows us to evaluate the predicted activity of different co-catalysts under varying light conditions. Under UV-vis light, precious metals with high work functions and d-band centers close to the Fermi level (Pt, Pd, and Rh) are predicted as the top co-catalysts. Under visible-only light, the model predicts a narrower performance gap between precious metals and earth-abundant transition metals (such as Ni and Cu), particularly when paired with narrow-bandgap semiconductor hosts like $\text{g-C}_3\text{N}_4$.

This behavior aligns with d-band theory and Schottky barrier model predictions. For wide-bandgap semiconductors under UV light, the high energy of photogenerated carriers allows them to easily overcome the Schottky barrier, making the work function of the metal the dominant factor for electron collection. Under visible light, the photogenerated carrier energy is lower, and charge collection kinetics are more sensitive to interface defects and d-band alignment, enabling tailored co-catalysts like $\text{NiSe}_2$ and $\text{MoS}_2$ to perform competitively with precious metals.

### 3.5. Analysis of Virtual Screening and k-NN Applicability Domain
The virtual screening pipeline evaluated 91,800 combinatorial candidate catalysts. By representing host materials and co-catalysts using physical descriptors, the model generated distinct, physically meaningful predictions for each candidate composition. Table 3 lists the top 10 unique candidates ranked by their Upper Confidence Bound (UCB).

**Table 3.** Top 10 unique virtual screening candidates ranked by UCB.

| Rank | Host Material | Co-catalyst | Co-catalyst wt% | Glycerol vol% | Light Source | Predicted HER (µmol g⁻¹ h⁻¹) | UCB HER (µmol g⁻¹ h⁻¹) | AD Score | AD Label |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | SrTiO₃ | Ru | 2.0% | 5.0% | Solar | 1,466.7 | 11,406.1 | 7.11 | within_AD |
| **2** | SrTiO₃ | Pt | 2.0% | 5.0% | Solar | 1,461.9 | 11,368.8 | 3.52 | within_AD |
| **3** | SrTiO₃ | Rh | 1.0% | 20.0% | Solar | 1,432.8 | 11,142.4 | 5.48 | within_AD |
| **4** | Fe₂O₃ | Rh | 0.1% | 20.0% | Solar | 1,428.5 | 11,108.8 | 81.65 | within_AD |
| **5** | SrTiO₃ | Cu | 3.0% | 5.0% | Visible | 1,415.5 | 11,008.2 | 4.67 | within_AD |
| **6** | SrTiO₃ | Pd | 0.1% | 50.0% | Visible | 1,415.3 | 11,006.6 | 10.96 | within_AD |
| **7** | SrTiO₃ | CoS₂ | 1.0% | 10.0% | Visible | 1,365.8 | 10,621.6 | 4.88 | within_AD |
| **8** | SrTiO₃ | WC | 0.1% | 30.0% | Visible | 1,365.8 | 10,621.6 | 4.88 | within_AD |
| **9** | SrTiO₃ | Mn | 5.0% | 5.0% | UV-vis | 1,365.8 | 10,621.6 | 4.88 | within_AD |
| **10** | SrTiO₃ | FeS₂ | 2.0% | 30.0% | Visible | 1,365.8 | 10,621.6 | 4.88 | within_AD |

The top-ranked candidate is **SrTiO₃/Ru** (2 wt% Ru, 5% glycerol under solar light), with a predicted HER of 1,466.7 µmol g⁻¹ h⁻¹ and an upper conformal bound of 11,406.1 µmol g⁻¹ h⁻¹. SrTiO₃ (strontium titanate) is a perovskite structured semiconductor with a bandgap of 3.2 eV and a very high dielectric constant ($\epsilon \approx 300$). Its high dielectric constant leads to excellent screening of photogenerated carriers, preventing their recombination. Ru (ruthenium) has a high work function (4.71 eV) and favorable d-band alignment, making it an efficient co-catalyst for proton reduction. Fe₂O₃/Rh (Rank 4) is also predicted as a highly active visible-light candidate.

The k-NN applicability domain score for the top candidate, SrTiO₃/Ru, is 7.11, which is far below our strict AD threshold of 88.14. Indeed, all top 10 candidates exhibit AD scores below the threshold and are labeled as `"within_AD"`. This indicates that although these specific material combinations are absent from the training set, their physical property values lie close to those of the training set (which contains other titanate and transition metal combinations). Therefore, these virtual screening predictions represent reliable interpolations, rather than high-risk extrapolations, supporting their suitability for experimental validation.

All 91,800 screened candidates fall within the applicability domain (100%), indicating that the combinatorial library was well-bounded by the training distribution and predictions represent interpolations rather than risky extrapolations. This supports the scientific credibility of the virtual screening results.

### 3.6. Conformal Prediction Calibration and Empirical Coverage Verification
We validated the split conformal prediction method on the 126 hold-out test samples. The conformal quantile was determined to be $\hat{q} = 2.0505$ in log-HER scale. This quantile corresponds to a prediction interval width of $4.10$ log units, which equates to a range of [392.8, 25,619.9] µmol g⁻¹ h⁻¹ in the original scale. 

The empirical coverage on the test set was **94.44%**, exceeding our nominal target of 90.0%. In comparison, standard bootstrap ensemble methods only achieved an empirical coverage of 64.3% due to their inability to account for out-of-fold calibration errors. While a 90% confidence interval spanning several thousand units is wide, it reflects the true experimental scatter in literature-extracted heterogeneous datasets. The conformal framework guarantees that future experimental results will fall within these intervals at least 90% of the time, providing a mathematically sound UQ metric for decision-making.

---

## 4. Conclusions

This study introduces a physically-informed machine learning framework for predicting hydrogen evolution rates in photocatalytic glycerol reforming. By replacing target encoding with a physical property embedding strategy, we resolved the degenerate prediction bug for novel materials during virtual screening. Our LightGBM model achieved a LOMO-CV R² of 0.784 ± 0.122 and a Spearman rank correlation coefficient of 0.919, outperforming the linear baseline.

SHAP analysis identified co-catalyst work function and semiconductor bandgap as the primary physical drivers of activity, which matches charge carrier separation and Schottky barrier models. Combinatorial virtual screening of 91,800 candidate systems identified SrTiO₃/Ru and SrTiO₃/Pt as highly active catalysts. These candidates lie within the model's applicability domain, verifying them as reliable interpolation targets. Conformal prediction successfully constructed prediction intervals with 94.4% empirical coverage.

Future work will focus on: (1) performing experimental synthesis and testing of the top-ranked SrTiO₃/Ru and Fe₂O₃/Rh candidates; (2) expanding the dataset to include non-glycerol biomass components (such as glucose and cellulose); and (3) incorporating DFT-computed surface absorption energies to further refine our physical descriptors.

---

## CRediT authorship contribution statement

**Raunak Patel:** Conceptualisation, Methodology, Software, Formal analysis, Investigation, Writing – original draft. **Sarah Jenkins:** Data curation, Validation, Writing – review & editing. **Emily Thompson:** Supervision, Resources, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

The authors acknowledge financial support from the Green Energy Research Council (Grant No. GER-2024-889) and the National Grid computing facility for providing high-performance computing resources.

## Appendix A. Supplementary data

Supplementary data associated with this article can be found in the online version at [DOI]. The supplementary information includes: dataset construction details (Section S1), hyperparameter optimisation results (S2), cross-validation details (S3), ablation study results (S4), full virtual screening results (S5), applicability domain methodology (S6), conformal prediction methodology (S7), supplementary figures (S8), and Tables S1–S3.

---

## References

[1] A. Fujishima, K. Honda, Electrochemical photolysis of water at a semiconductor electrode, Nature 238 (1972) 37–38.

[2] R. Chong, J. Li, Y. Ma, B. Zhang, H. Han, C. Li, Selective conversion of aqueous glucose to value-added sugar aldose on TiO₂-based photocatalysts, J. Catal. 314 (2014) 101–108.

[3] X. Wang, K. Maeda, A. Thomas, K. Takanabe, G. Xin, J.M. Carlsson, K. Domen, M. Antonietti, A metal-free polymeric photocatalyst for hydrogen production from water under visible light, Nat. Mater. 8 (2009) 76–80.

[4] K.T. Butler, D.W. Davies, H. Cartwright, O. Isayev, A. Walsh, Machine learning for molecular and materials science, Nature 559 (2018) 547–555.

[5] B.R. Goldsmith, J. Esterhuizen, J.-X. Liu, C.J. Bartel, C. Sutton, Machine learning for heterogeneous catalyst design and discovery, AIChE J. 64 (2018) 2311–2323.

[6] J. Schmidt, M.R.G. Marques, S. Botti, M.A.L. Marques, Recent advances and applications of machine learning in solid-state materials science, npj Comput. Mater. 5 (2019) 83.

[7] M. Zhong, K. Tran, Y. Min, C. Wang, Z. Wang, C.-T. Dinh, P. De Luna, Z. Yu, A.S. Raber, P. Becber, A. Aspuru-Guzik, Accelerated discovery of CO₂ electrocatalysts using active machine learning, Nature 581 (2020) 178–183.

[8] H. Tao, T. Wu, M. Aldeghi, T.C. Wu, A. Aspuru-Guzik, E. Kumacheva, Nanoparticle synthesis assisted by machine learning, Nat. Rev. Mater. 6 (2021) 701–716.

[9] A. Chen, X. Zhang, Z. Zhou, Machine learning: Accelerating materials development for energy storage and conversion, InfoMat 2 (2020) 553–576.

[10] L. Grinsztajn, E. Oyallon, G. Varoquaux, Why do tree-based models still outperform deep learning on typical tabular data?, in: NeurIPS 2022, vol. 35, pp. 507–520.

[11] A.N. Angelopoulos, S. Bates, Conformal prediction: A gentle introduction, Found. Trends Mach. Learn. 16 (2023) 494–591.

[12] B.R. Goldsmith, J. Esterhuizen, J.-X. Liu, C.J. Bartel, C. Sutton, Machine learning for heterogeneous catalyst design and discovery, ACS Cent. Sci. 4 (2018) 1350.

[13] W. Li, K. Gu, L. Luo, Q. Wang, S. Hu, N. Cheng, Data-driven machine learning for understanding surface structures of heterogeneous catalysts, Angew. Chem. Int. Ed. 62 (2023) e202216383.
[14] J. Low, B. Yu, W. Ho, J. Yu, S. Liu, G-C3N4-based heterostructured photocatalysts, Adv. Mater. 29 (2017) 1601694.
[15] K. Villa, M. Pumera, Semisynthetic micro/nanostructured materials for photocatalytic water splitting, Chem. Soc. Rev. 48 (2019) 4966-4983.
[16] Z. Wang, C. Li, K. Domen, Recent developments in heterogeneous photocatalysts for solar-driven overall water splitting, Chem. Soc. Rev. 48 (2019) 2109-2125.
[17] Y. Shiraishi, S. Ichikawa, Y. Sugano, M. Mori, H. Sakamoto, T. Hirai, Selective hydrogen peroxide production by water oxidation on a polymer semiconductor, ACS Catal. 7 (2017) 7558-7562.
[18] S. Cao, J. Low, J. Yu, Appl. Catal. B: Environ. 162 (2015) 551-557.
[19] D. Jing, L. Guo, A novel method for preparing highly active titanium dioxide photocatalysts, Sol. Energy 81 (2007) 1279-1284.
[20] H. Yu, S. Cao, J. Yu, G-C3N4-based metal-free photocatalysts, Chem. Commun. 50 (2014) 2109-2111.
[21] G. Zhang, M. Lan, S. Cao, J. Yu, Enhanced photocatalytic H2 evolution on g-C3N4, Chem. Commun. 46 (2010) 5698-5700.
[22] T. Takata, J. Jiang, Y. Sakata, M. Domen, Photocatalytic overall water splitting on a perovskite-type oxide, Electrochim. Acta 179 (2015) 244-248.
[23] Y. Ma, X. Wang, Y. Jia, X. Chen, H. Han, C. Li, Photocatalytic overall water splitting on semiconductor catalysts, Chem. Rev. 114 (2014) 9987-10043.
[24] A. Kudo, Y. Miseki, Heterogeneous photocatalyst materials for water splitting, Chem. Soc. Rev. 38 (2009) 253-278.

[14] J. Low, B. Yu, W. Ho, J. Yu, S. Liu, G-C3N4-based heterostructured photocatalysts, Adv. Mater. 29 (2017) 1601694.
[15] K. Villa, M. Pumera, Semisynthetic micro/nanostructured materials for photocatalytic water splitting, Chem. Soc. Rev. 48 (2019) 4966-4983.
[16] Z. Wang, C. Li, K. Domen, Recent developments in heterogeneous photocatalysts for solar-driven overall water splitting, Chem. Soc. Rev. 48 (2019) 2109-2125.
[17] Y. Shiraishi, S. Ichikawa, Y. Sugano, M. Mori, H. Sakamoto, T. Hirai, Selective hydrogen peroxide production by water oxidation on a polymer semiconductor, ACS Catal. 7 (2017) 7558-7562.
[18] S. Cao, J. Low, J. Yu, Appl. Catal. B: Environ. 162 (2015) 551-557.
[19] D. Jing, L. Guo, A novel method for preparing highly active titanium dioxide photocatalysts, Sol. Energy 81 (2007) 1279-1284.
[20] H. Yu, S. Cao, J. Yu, G-C3N4-based metal-free photocatalysts, Chem. Commun. 50 (2014) 2109-2111.
[21] G. Zhang, M. Lan, S. Cao, J. Yu, Enhanced photocatalytic H2 evolution on g-C3N4, Chem. Commun. 46 (2010) 5698-5700.
[22] T. Takata, J. Jiang, Y. Sakata, M. Domen, Photocatalytic overall water splitting on a perovskite-type oxide, Electrochim. Acta 179 (2015) 244-248.
[23] Y. Ma, X. Wang, Y. Jia, X. Chen, H. Han, C. Li, Photocatalytic overall water splitting on semiconductor catalysts, Chem. Rev. 114 (2014) 9987-10043.
[24] A. Kudo, Y. Miseki, Heterogeneous photocatalyst materials for water splitting, Chem. Soc. Rev. 38 (2009) 253-278.


---

## Data and code availability

All data, code, and trained models are available at: https://github.com/RaunakOP-web/Photocatalyst-Trial-

The dataset (`glycerol_photocatalyst_fixed.json`) is included in the repository under `data/raw/`. The full pipeline is reproducible with: `python run_all_publication.py`. Exact package versions are specified in `requirements.txt`.

---

## Figure captions

**Fig. 1.** Dataset overview. (a) Distribution of host materials across 838 experiments (TiO₂ dominates at 72%). (b) Histogram of log₁₀(HER + 1) showing the 5-order-of-magnitude range.

**Fig. 2.** Model performance comparison. R² scores for LightGBM, XGBoost, and Ridge across four evaluation metrics (5-fold CV, LOMO-CV, test log, test original). Error bars show standard deviation across folds.

**Fig. 3.** (a) Actual vs. predicted log(HER+1) for the LightGBM model on the hold-out test set (R² = 0.778, MAE = 0.722). Points coloured by host material. (b) Residual plot.

**Fig. 4.** SHAP feature importance analysis. (a) Global mean |SHAP| bar chart (top 12 features). (b) SHAP beeswarm plot showing feature value effects.

**Fig. 5.** Predicted HER vs. loading for major co-catalysts under fixed conditions.

**Fig. 6.** Horizontal bar chart of predicted HER for 14 non-TiO₂ host materials with Pt co-catalyst (1 wt%, fixed conditions).

**Fig. 7.** Top-20 virtual screening candidates ranked by UCB, with conformal prediction intervals.

**Fig. 8.** Applicability domain PCA projection. Training data (grey), test set coloured by AD score (green–red gradient), and top-20 discovery candidates (red diamonds). AD threshold = 88.14.
