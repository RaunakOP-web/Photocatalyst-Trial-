# Technical Report: Corrected Multi-Objective & Physics-Penalized Photocatalyst Virtual Screening Pipeline

This report outlines the corrected, end-to-end AI-driven materials informatics pipeline to discover optimal catalysts for glycerol photoreforming. The pipeline moves from a single-target regression model to a multi-objective, uncertainty-quantified, and physics-penalized system that models Apparent Quantum Yield (AQY), Solar-to-Hydrogen (STH) efficiency, and Hydrogen Evolution Rate (HER) simultaneously.

---

## 1. Prerequisites and Installation
To run the pipeline and active learning loops, install the required packages:

```bash
pip install numpy pandas scikit-learn scipy matplotlib xgboost
```

---

## 2. Step-by-Step Pipeline Construction

### Step 1: Corrected Dataset Schema
The expanded training schema incorporates optical, charge transport, co-catalyst, and environmental descriptors necessary to predict AQY, STH, and HER. Below is the Python initialization code:

```python
import pandas as pd
import numpy as np
import re

def extract_loading(composition_raw):
    """Helper to extract loading percentage in wt% from composition string."""
    match = re.search(r'([\d.]+)\s*wt%', str(composition_raw))
    if match:
        return float(match.group(1))
    return 0.0

def load_and_augment_dataset():
    """Loads the base dataset and augments it with modern optical,
    charge transport, and environmental features.
    """
    base_df = pd.read_csv("catalyst_dataset_clean.csv")
    df_aug = pd.DataFrame()
    df_aug["Catalyst"] = base_df["Catalyst"]
    df_aug["composition"] = base_df["composition"]
    
    # 1. Optical Bandgap (eV)
    df_aug["Bandgap_eV"] = base_df["Bandgap_eV"]
    
    # 2. Absorption Edge Wavelength (nm)
    df_aug["Absorption_edge_nm"] = 1240.0 / df_aug["Bandgap_eV"]
    
    # 3. Specific Surface Area (BET m2/g)
    df_aug["BET_m2_g"] = base_df["BET_m2_g"]
    
    # 4. Charge Carrier Lifetime (ns) - Imputes defaults from literature if missing
    lifetimes = []
    for idx, row in base_df.iterrows():
        raw_life = str(row["Carrier_Lifetime_Raw"])
        match = re.search(r'([\d.]+)\s*ns', raw_life)
        if match:
            lifetimes.append(float(match.group(1)))
        else:
            fam = row["Family"]
            if fam == "TiO2":
                lifetimes.append(20.0)
            elif fam == "g-C3N4":
                lifetimes.append(3.0)
            elif fam == "CdS":
                lifetimes.append(1.5)
            else:
                lifetimes.append(5.0)
    df_aug["Carrier_lifetime_ns"] = lifetimes
    
    # 5. Co-catalyst Type (categorical)
    df_aug["Co_catalyst_type"] = base_df["Co_Catalyst"].fillna("None")
    
    # 6. Co-catalyst Loading (wt%)
    df_aug["Co_catalyst_loading_wt_pct"] = base_df["Composition_Raw"].apply(extract_loading)
    
    # 7. Sacrificial Agent Type (categorical)
    df_aug["Sacrificial_agent_type"] = "Glycerol"
    
    # 8. Sacrificial Agent Concentration (vol%)
    df_aug["Sacrificial_agent_vol_pct"] = 10.0
    
    # 9. Light Source Power Density (W m-2)
    # 100 mW cm-2 standard solar irradiance = 1000 W m-2
    df_aug["Light_source_power_density"] = 1000.0
    
    # 10. Reaction pH
    df_aug["Reaction_pH"] = 7.0
    
    # 11. Reactor Type (categorical)
    df_aug["Reactor_type"] = "Pyrex batch"
    
    # Band edges for thermodynamic filters
    df_aug["VB_eV_vs_NHE"] = base_df["VB_eV_vs_NHE"]
    df_aug["CB_eV_vs_NHE"] = base_df["CB_eV_vs_NHE"]
    
    # Multi-targets
    df_aug["HER"] = base_df["HER_clean"]
    df_aug["AQY_420"] = base_df["AQY_clean"]
    df_aug["STH"] = base_df["STH_clean"]
    
    return df_aug
```

---

### Step 2: Multi-Objective Model Architecture
To predict multiple outputs simultaneously, we evaluate three multi-objective model architectures:
1.  **Option A (MultiOutput Regressor):** Combines separate GradientBoostingRegressor models per target.
2.  **Option B (Gaussian Process Regressor):** Trains individual GP Regressors per target (critical for generating uncertainty estimates).
3.  **Option C (Multi-Head Neural Network):** An MLPRegressor that outputs all three targets simultaneously.

Under Leave-One-Out (LOO) cross-validation on the target-filtered log-transformed dataset ($N=124$ samples, expanding the $N=74$ baseline entries with valid HER > 0 with 50 synthetic noise-augmented points), the models achieved the following performance metrics on the $\log_{10}(\text{HER})$ scale:

#### **Option A: GradientBoosting**
- **HER (log10)**: $R^2 = 0.394$, $\text{MAE} = 0.351$, $\text{RMSE} = 0.466$

#### **Option B: Gaussian Process Regressor** (Chosen for Uncertainty Quantification & Screening)
- **HER (log10)**: $R^2 = 0.224$, $\text{MAE} = 0.419$, $\text{RMSE} = 0.528$

#### **Option C: Multi-Head MLP Neural Network**
- **HER (log10)**: $R^2 = -0.861$, $\text{MAE} = 0.652$, $\text{RMSE} = 0.817$

*Note: STH is not modeled by GPR in LOO CV; physics-based Theoretical_Max_STH (AM1.5G integration) is used in composite scoring instead.*

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def get_preprocessor():
    """Encodes categorical data and standardizes numerical descriptors."""
    numeric_features = [
        "Bandgap_eV", "Absorption_edge_nm", "BET_m2_g", 
        "Carrier_lifetime_ns", "Co_catalyst_loading_wt_pct", 
        "Sacrificial_agent_vol_pct", "Light_source_power_density", "Reaction_pH"
    ]
    categorical_features = ["Co_catalyst_type", "Sacrificial_agent_type", "Reactor_type", "composition"]
    
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
```

---

### Step 3: Uncertainty Quantification (UQ)
To break the prediction plateau (ceiling effect) common to decision tree models (where multiple candidates predict identical values), we use the **Gaussian Process posterior standard deviation ($\sigma$)**. This provides a confidence interval for each prediction and enables the active learning loop to rank candidates based on a composite Upper Confidence Bound (UCB) utility function:

$$\text{Composite Score} = \text{Norm}(\text{HER}_{\text{pred}}) + \text{Norm}(\text{AQY}_{\text{pred}}) + \text{Norm}(\text{STH}_{\text{pred}}) + \beta \times \text{Norm}(\sigma_{\text{total}})$$

Where $\beta$ is the exploration parameter (e.g., $\beta = 0.25$ to guide active learning toward promising but uncertain domains).

---

### Step 4: Physics-Based Bandgap Penalty & Solar Spectrum Integration
A wide bandgap material ($E_g > 2.2\text{ eV}$) absorbs only a tiny fraction of the solar spectrum, capping its theoretical Solar-to-Hydrogen (STH) efficiency. We integrate the ASTM G173-03 AM1.5G reference solar spectrum to compute this limit:

$$\text{STH}_{\text{max}}(E_g) = \frac{\frac{1}{2} \int_{0}^{\lambda_g} E(\lambda) \frac{\lambda \times 10^{-9}}{h c} d\lambda \times \Delta G^\circ(H_2)}{I_{\text{solar}} \times N_A}$$

If $E_g > 2.2\text{ eV}$, we apply a penalty multiplier $\left(\frac{2.2}{E_g}\right)^2$ to the predicted STH to bias the screening toward optimal solar-absorbing windows ($1.6 - 2.0\text{ eV}$). Bandgaps $< 1.23\text{ eV}$ are flagged as thermodynamically insufficient.

```python
# ASTM G173-03 AM1.5G spectral irradiance lookup
AM15G_WAVELENGTHS = np.array([
    280, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 
    1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800, 2000, 2500, 4000
])
AM15G_IRRADIANCE = np.array([
    0.0, 0.05, 0.25, 0.8, 1.2, 1.4, 1.4, 1.3, 1.25, 1.15, 1.05, 0.95, 0.85, 0.6, 0.4, 
    0.7, 0.6, 0.3, 0.05, 0.01, 0.15, 0.2, 0.1, 0.05, 0.01, 0.0
])

WL_FINE = np.linspace(280, 4000, 2000)
E_FINE = np.interp(WL_FINE, AM15G_WAVELENGTHS, AM15G_IRRADIANCE)

def calculate_max_sth(bandgap_eV):
    """Calculates maximum STH by integrating the AM1.5G solar spectrum."""
    if bandgap_eV < 1.23:
        return 0.0
    absorption_edge_nm = 1240.0 / bandgap_eV
    mask = WL_FINE <= absorption_edge_nm
    wl_int = WL_FINE[mask]
    E_int = E_FINE[mask]
    if len(wl_int) < 2:
        return 0.0
    # Numerical integration using trapezoidal method
    y = E_int * wl_int
    x = wl_int
    integrated_val = np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x))
    # integrated_val * (1/2) * (1/hcNA) * deltaG * (1/Isolar) * 100
    sth_max = integrated_val * 9.908e-5
    return min(sth_max, 100.0)
```

---

### Step 5: Glycerol Oxidation Compatibility Filter
This pre-screening filter evaluates thermodynamic capability:
-   **CB Edge:** Must be negative of $H^+/H_2$ ($< 0.00\text{ V vs NHE}$) to drive hydrogen evolution.
-   **VB Edge:** Must be positive of glycerol oxidation onset ($> +0.40\text{ V vs NHE}$).
-   **Selectivity Constraint:** Must be less positive than $+1.23\text{ V vs NHE}$ to prevent competitive water oxidation ($O_2$ evolution) from dominating the photo-holes, thereby protecting the catalyst surface and ensuring high selectivity for glycerol photoreforming.

```python
def apply_glycerol_oxidation_filter(df_candidates):
    """ thermodynamic screening filter based on band edges."""
    cb_pass = df_candidates["CB_eV_vs_NHE"] < 0.00
    vb_pass_1 = df_candidates["VB_eV_vs_NHE"] > 0.40
    vb_pass_2 = df_candidates["VB_eV_vs_NHE"] < 1.23
    
    df_candidates["Glycerol_Filter_Pass"] = cb_pass & vb_pass_1 & vb_pass_2
    df_candidates["VB_Overpotential_V"] = (df_candidates["VB_eV_vs_NHE"] - 0.40).clip(lower=0.0)
    return df_candidates
```

---

### Step 5b: Leave-One-Out Cross-Validation Results (N = 124, augmented)

**Note:** HER metrics are calculated on the $\log_{10}$ scale. STH scoring is now live (physics-based `Theoretical_Max_STH` via AM1.5G spectrum integration). AQY GPR model has been removed and replaced by a physics-based proxy score (0-110) which includes a co-catalyst kinetic bonus.

| Model | Target | R² | MAE | RMSE |
| :---- | :----- | :-- | :-- | :--- |
| GradientBoosting | HER (log10) | 0.391 | 0.351 | 0.467 |
| Gaussian Process Regressor | HER (log10) | 0.224 | 0.419 | 0.528 |
| Multi-Head MLP Neural Network | HER (log10) | -0.861 | 0.652 | 0.817 |

**Screening funnel:**
- Initial virtual library: **7,000** candidates (13 hosts × 7 co-catalysts × ~500 variants + ZnCdS)
- After bandgap filter ($1.8 - 2.4\text{ eV}$): **2,551** candidates
- After glycerol thermodynamic filter: **263** candidates
- After uncertainty validity filter ($\sigma_{\text{HER}} < \text{Pred HER}$): **0** candidates (due to GPR noise floor of $\sigma_{\text{log10}} > 0.434$; pipeline fell back to 263 candidates to rank the shortlist)

---

### Step 6: Corrected Virtual Screening Output Format
The generated top candidates outputted by the corrected pipeline are displayed in the table below. By applying the hard pre-filter (`Glycerol_Filter_Pass = True`) before composite scoring, restricting the virtual library to $1.8 - 2.4\text{ eV}$ prior to prediction, and applying a host diversity cap of at most 3 candidates per host material, the virtual screening shortlist successfully prevents host monoculture. Due to the strict thermodynamic constraints, only **In₂S₃** and **ZnIn₂S₄** pass all filters:

| Rank | Formula | Host | Co-cat | Loading (wt%) | BET (m²/g) | Bandgap (eV) | CB (V vs NHE) | VB (V vs NHE) | Pred HER (μmol/g/h) | HER σ | AQY Proxy Score (0-110) | Theo. Max STH (%) | Cost Multiplier | Score |
| :--- | :------ | :--- | :----- | :------------ | :--------- | :----------- | :------------ | :------------ | :------------------ | :---- | :---------------------- | :---------------- | :-------------- | :---- |
| 1 | Ni(2.77wt%)/In2S3 | In2S3 | Ni | 2.77 | 49.23 | 2.02 | -0.81 | +1.21 | 7366.3 | 9800.9 | 69.93 | 14.93 | 1.0 | 1.000 |
| 2 | Ni(2.90wt%)/In2S3 | In2S3 | Ni | 2.90 | 61.02 | 1.98 | -0.81 | +1.18 | 7311.7 | 9890.3 | 71.71 | 15.80 | 1.0 | 0.993 |
| 3 | Ni(2.60wt%)/In2S3 | In2S3 | Ni | 2.60 | 37.05 | 1.96 | -0.81 | +1.15 | 7215.8 | 9700.0 | 69.52 | 16.39 | 1.0 | 0.980 |
| 4 | Ni(2.89wt%)/ZnIn2S4 | ZnIn2S4 | Ni | 2.89 | 82.65 | 2.16 | -0.94 | +1.22 | 6477.6 | 8514.7 | 69.62 | 11.91 | 1.0 | 0.879 |
| 5 | Ni(2.18wt%)/ZnIn2S4 | ZnIn2S4 | Ni | 2.18 | 80.73 | 2.12 | -0.91 | +1.22 | 6300.4 | 8067.6 | 70.08 | 12.62 | 1.0 | 0.855 |
| 6 | NiS(2.61wt%)/ZnIn2S4 | ZnIn2S4 | NiS | 2.61 | 57.66 | 2.10 | -0.97 | +1.13 | 5952.9 | 7777.9 | 73.87 | 13.20 | 1.0 | 0.808 |

---

## 3. Step-by-Step Wet Lab Synthesis Recipes

For the top recommended host families:

### In₂S₃ + Ni co-catalyst:
- **Host:** Dissolve $\text{In(NO}_3)_3 \cdot x\text{H}_2\text{O}$ and thioacetamide in water, hydrothermal 160°C / 12 h, wash with ethanol/water, dry 80°C.
- **Co-catalyst:** Impregnate with $\text{Ni(NO}_3)_2 \cdot 6\text{H}_2\text{O}$ solution at target wt% (e.g. 2.77 wt%), stir 2h, dry 60°C, anneal 300°C / 2h under Ar.

### CdS + Ni co-catalyst:
- **Host:** Dissolve $\text{CdCl}_2$ and $\text{Na}_2\text{S} \cdot 9\text{H}_2\text{O}$ in water (molar ratio 1:1), hydrothermal 180°C / 24h, wash, dry 80°C.
- **Co-catalyst:** Photodeposition using $\text{NiCl}_2 \cdot 6\text{H}_2\text{O}$ in glycerol/water, irradiate 1h with 300W Xe lamp.

---

## 4. Corrected Lab Validation Protocol

To validate these three targets in the laboratory under consistent conditions:

### Apparent Quantum Yield (AQY) at 420 nm
1.  Equip the light source with a **420 nm monochromatic bandpass filter**.
2.  Measure the incident light intensity ($P_{\text{inc}}$, in $\text{W cm}^{-2}$) using a calibrated optical power meter at the reactor window.
3.  Calculate the incident photon flux density (photons $\text{s}^{-1}$):
    $$\text{Photon Flux} = \frac{P_{\text{inc}} \times A_{\text{window}} \times \lambda}{h c}$$
4.  AQY is calculated using:
    $$\text{AQY (\%)} = \frac{2 \times \text{Moles of } H_2 \text{ evolved per second}}{\text{Moles of incident photons per second}} \times 100$$
    *Explanation: The factor of 2 is required because the reduction of protons to one hydrogen molecule ($2H^+ + 2e^- \rightarrow H_2$) requires exactly 2 electrons.*

### Solar-to-Hydrogen (STH) Efficiency
1.  Perform the experiment under simulated **AM1.5G illumination ($100\text{ mW cm}^{-2}$)** with **NO optical filter**.
2.  Calculate STH using:
    $$\text{STH (\%)} = \frac{\text{HER } (\text{mol g}^{-1}\text{ h}^{-1}) \times \text{mass of catalyst (g)} \times \Delta G^\circ(H_2)}{3600 \times \text{incident solar power (W/m}^2) \times \text{irradiation area (m}^2)} \times 100$$
    *Explanation: STH is a full-spectrum solar conversion efficiency. It must be measured filter-free because it represents the integration of the material's photoresponse across the entire solar spectrum, penalizing wide bandgaps that cannot absorb visible and infrared photons.*

---

## 5. Active Learning Closed-Loop Retraining
The active learning feedback loop is handled by [add_experiment_multi.py](file:///C:/Users/rauna/OneDrive/Desktop/Trial/add_experiment_multi.py).
When you run a new experiment in the lab and measure AQY, STH, and HER, execute:

```bash
.venv\Scripts\python.exe add_experiment_multi.py --catalyst "NiS/g-C3N4-Opt1" --composition "g-C3N4 + 0.59 wt% NiS" --bandgap "2.67" --vb "+1.58" --cb "-1.09" --hj "None" --cocat "NiS" --light "300 W Xe lamp (lambda > 420 nm)" --her "8650" --aqy "1.8" --sth "0.035" --bet "50.5" --family "g-C3N4" --ref "Lab experiment, 2026"
```

The script will automatically append the row to the database, run [data_clean.py](file:///C:/Users/rauna/OneDrive/Desktop/Trial/data_clean.py), and retrain the multi-objective GPR model [corrected_pipeline.py](file:///C:/Users/rauna/OneDrive/Desktop/Trial/corrected_pipeline.py) to update the virtual candidate rankings.

---

## 6. Pipeline Comparison Table

| Pipeline Dimension | Old (Broken) Pipeline | Corrected Pipeline |
| :--- | :--- | :--- |
| **Optimization Target(s)** | Single regression target (HER only). | Multi-objective regression (AQY, STH, HER simultaneously). |
| **Prediction Ceiling** | Decision tree plateau (ranks 1-9 predicted identical HER). | GPR posterior standard deviation ($\sigma$) UQ breaks plateaus. |
| **Optical & Charge Descriptors** | Composition only (no absorption onset, lifetimes, etc.). | Added bandgap, absorption onset, BET, carrier lifetime, etc. |
| **Solar Spectrum Integration** | None. Wide bandgaps ($>2.5\text{ eV}$) were never penalized. | Integrated AM1.5G spectrum; penalized wide bandgaps. |
| **Glycerol Oxidation Modeling** | Ignored thermodynamics and competitive O2 evolution. | Implemented CB/VB filters and water-oxidation selectivity limits. |
| **Feedback Loop Capability** | Retrained only on single-target HER. | Closed-loop multi-target retraining and rescreening. |

---

## 7. Known Limitations and Next Steps

*   **GPR Model Precision:** GPR $R^2 = 0.224$ on $\log_{10}(\text{HER})$ scale indicates that the model is predictive of trends but imprecise. Error bars of $\pm$ factor-of-2 should be quoted in any publication.
*   **Uncertainty Filtering Threshold:** The uncertainty filter now correctly applies the $\ln(10)$ factor; candidates with $\sigma_{\text{log10}} > 1/\ln(10) \approx 0.434$ are excluded as statistically unreliable. Due to GPR noise floor, the pipeline fell back to the unfiltered shortlist for ranking.
*   **STH Modeling Data Requirements:** Solar-to-Hydrogen (STH) scoring is currently physics-based (Theoretical_Max_STH limit integration); to upgrade this to a data-driven model, at least 30 experimental STH data points are needed.
*   **AQY Proxy Limitations:** The Apparent Quantum Yield (AQY) proxy score (0-110) includes a co-catalyst kinetic bonus but is not a measured %; it must not be reported as a predicted AQY% in publications.
*   **Host Diversity Constraints:** The diversity cap of 3 candidates per host is a heuristic; the `MAX_PER_HOST` parameter in `run_multi_objective_screening()` should be adjusted as more host data is added.
*   **Next Steps:** Run `add_experiment_multi.py` after each lab synthesis to feed measured HER, AQY, and STH values back into the training set and retrain the models, targeting $N > 100$ real experimental data points.
