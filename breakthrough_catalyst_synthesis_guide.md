# Actionable Laboratory Execution Report: Top 10 Optimized Photocatalysts

This report summarizes the virtual screening results of the top 10 optimized catalyst candidates, details their material science core analysis, provides step-by-step wet lab synthesis recipes, and outlines the standard testing protocol to enable experimental validation and closed-loop retraining.

---

## Phase 1: Data Extraction & Structured Presentation

The following Python script was executed to extract the candidates from [screened_top_10_catalysts.csv](file:///C:/Users/rauna/OneDrive/Desktop/Trial/screened_top_10_catalysts.csv):

```python
import pandas as pd
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    csv_path = "screened_top_10_catalysts.csv"
    df = pd.read_csv(csv_path)
    
    print("| Rank | Catalyst/Composition Formula | Predicted HER (µmol g⁻¹ h⁻¹) | Predicted Bandgap (eV) | Calculated CB (eV vs NHE) | Calculated VB (eV vs NHE) | Specific Surface Area (BET m² g⁻¹) | Co-catalyst | Loading (wt%) |")
    print("|------|------------------------------|------------------------------|------------------------|---------------------------|---------------------------|-------------------------------------|-------------|---------------|")
    
    for i, row in df.iterrows():
        rank = i + 1
        comp = row['composition']
        her = f"{row['HER_clean predicted']:.1f}"
        eg = f"{row['Bandgap_eV']:.2f}"
        cb = f"{row['CB_eV_vs_NHE']:.2f}"
        vb = f"{row['VB_eV_vs_NHE']:.2f}"
        bet = f"{row['BET_m2_g']:.1f}"
        cocat = row['co_catalyst']
        w = f"{row['loading_wt_pct']:.2f}"
        print(f"| {rank} | {comp} | {her} | {eg} | {cb} | {vb} | {bet} | {cocat} | {w} |")

if __name__ == "__main__":
    main()
```

### Top 10 Screened Catalyst Candidates

| Rank | Catalyst/Composition Formula | Predicted HER (µmol g⁻¹ h⁻¹) | Predicted Bandgap (eV) | Calculated CB (eV vs NHE) | Calculated VB (eV vs NHE) | Specific Surface Area (BET m² g⁻¹) | Co-catalyst | Loading (wt%) |
| :--- | :--------------------------- | :--------------------------- | :--------------------- | :------------------------ | :------------------------ | :--------------------------------- | :---------- | :------------ |
| 1    | Ni0.006029S0.006029C3N4      | 8729.6                       | 2.67                   | -1.09                     | 1.58                      | 50.5                               | NiS         | 0.59          |
| 2    | Ni0.010271S0.010271C3N4      | 8729.6                       | 2.65                   | -1.08                     | 1.57                      | 53.8                               | NiS         | 1.00          |
| 3    | Ni0.008811S0.008811C3N4      | 8729.6                       | 2.66                   | -1.08                     | 1.57                      | 44.4                               | NiS         | 0.86          |
| 4    | Ni0.009571S0.009571C3N4      | 8729.6                       | 2.65                   | -1.08                     | 1.57                      | 29.3                               | NiS         | 0.93          |
| 5    | Ni0.004624S0.004624C3N4      | 8729.6                       | 2.68                   | -1.09                     | 1.59                      | 52.8                               | NiS         | 0.45          |
| 6    | Ni0.005701S0.005701C3N4      | 8729.6                       | 2.67                   | -1.09                     | 1.58                      | 39.7                               | NiS         | 0.56          |
| 7    | Ni0.003739S0.003739C3N4      | 8729.6                       | 2.68                   | -1.09                     | 1.59                      | 66.0                               | NiS         | 0.37          |
| 8    | Ni0.006427S0.006427C3N4      | 8729.6                       | 2.67                   | -1.09                     | 1.58                      | 53.4                               | NiS         | 0.63          |
| 9    | Ni0.011735S0.011735C3N4      | 8729.6                       | 2.64                   | -1.08                     | 1.57                      | 54.4                               | NiS         | 1.14          |
| 10   | Ni0.039073C3N4               | 8535.4                       | 2.58                   | -1.05                     | 1.53                      | 36.6                               | Ni          | 2.43          |

---

## Phase 2: Chemical Core Analysis

### 1. AutoML Prioritization of $g-C_3N_4$ / $NiS$ Host-Guest Configuration
The AutoML pipeline prioritized the transition-metal-sulfide-loaded graphitic carbon nitride ($NiS$/$g-C_3N_4$) configuration over traditional precious metal systems ($Pt$) due to several underlying physical and descriptor-based synergies:

*   **Atomic Coordination and Dispersion:** The heptazine ring cavity structure of $g-C_3N_4$ is rich in nitrogen lone-pair electrons. These local environments form exceptionally strong coordination bonds with $Ni^{2+}$ ions. During in-situ synthesis, this promotes atomic-scale dispersion of nickel co-catalysts, preventing bulk metal agglomeration which commonly degrades $Pt$-loaded systems.
*   **Schottky/Heterojunction Energetics:** $NiS$ acts as a metallic-like or narrow-bandgap semiconductor co-catalyst. Contact with the n-type $g-C_3N_4$ host forms a highly efficient charge transfer interface (often modeled as an S-scheme or Schottky barrier). The built-in electric field drives photogenerated electrons from the $g-C_3N_4$ conduction band directly into the $NiS$ active sites, accelerating the kinetics of proton reduction.
*   **Organic Poisoning Resistance in Glycerol Media:** While $Pt$ is the premier hydrogen evolution catalyst in pure water, it suffers from severe poisoning in organic-rich sacrificial media (such as glycerol). Partially oxidized carbonaceous intermediates (e.g., $CO$-like species, aldehydes) adsorb strongly to the highly active $Pt$ surfaces, blocking catalytic active sites. Conversely, transition metal sulfides like $NiS$ exhibit much weaker chemisorption energies for carbonaceous intermediates, maintaining clean active surfaces and stable long-term hydrogen production rates.
*   **Atypical Descriptor Combinations:** Feature extraction via Magpie maps properties such as d-electron abundance, electronegativity differences, and coordinate geometry. The combination of carbon/nitrogen elemental traits ($C_3N_4$) and soft-anion transition metal chalcogenides ($Ni-S$) creates a highly synergistic multidimensional feature representation that the ML models identify as optimal for sacrificial-donor-driven photocatalysis.

### 2. Thermodynamic Edge Alignment for HER and GOR
The engineered band-edge potentials of the top candidates are perfectly tuned to drive both reduction and oxidation reactions simultaneously:

```
Energy (vs NHE)
  ^
  |  -1.5 V --
  |            \   CB Edge (Predicted: -1.08 V to -1.09 V vs NHE)
  |  -1.0 V -----===========================================
  |            \
  |  -0.5 V --  \  Thermodynamic H+/H2 Reduction (pH 7: -0.41 V vs NHE)
  |              -------------------------------------------
  |   0.0 V -----  Thermodynamic H+/H2 Reduction (pH 0: 0.0 V vs NHE)
  |            /
  |  +0.5 V --  /  Glycerol Oxidation Reaction (GOR) (+0.4 V to +0.8 V vs NHE)
  |            / -------------------------------------------
  |  +1.0 V --/
  |            \
  |  +1.5 V -----===========================================
  |            /   VB Edge (Predicted: +1.57 V to +1.59 V vs NHE)
  |  +2.0 V --
```

*   **Conduction Band (CB) Reduction Force:** The reduction potential for $H^+/H_2$ is $0.0\text{ V vs NHE}$ at pH 0, shifting to $-0.41\text{ V}$ at pH 7. The predicted CB positions of the candidates (**$-1.08\text{ V}$ to $-1.09\text{ V vs NHE}$**) provide a massive thermodynamic driving force (overpotential of **$\sim 0.68\text{ V}$** under neutral conditions) for reducing protons to hydrogen gas.
*   **Valence Band (VB) Oxidation Force:** The thermodynamic potential required for the Glycerol Oxidation Reaction (GOR) to form value-added organic products (like glyceraldehyde or dihydroxyacetone) is in the range of **$+0.4\text{ V}$ to $+0.8\text{ V vs NHE}$**. The predicted VB positions of the candidates (**$+1.57\text{ V}$ to $+1.59\text{ V vs NHE}$**) provide a high thermodynamic overpotential (**$> 0.8\text{ V}$**) to drive hole-induced glycerol oxidation. Crucially, the VB potential is positive enough to easily oxidize glycerol but remains significantly less positive than the potential required for water oxidation (Oxygen Evolution Reaction, $+1.23\text{ V}$ at pH 0 / $+0.82\text{ V}$ at pH 7), steering the reaction pathway away from kinetically sluggish oxygen generation and toward value-added organic oxidation.

---

## Phase 3: Step-by-Step Wet Lab Synthesis Recipes

The recipes below outline a scalable two-step thermal/hydrothermal route to prepare a standard **1.0-gram final composite batch** for the top 3 highest-performing candidates.

### 1. Precursor Reagents
*   **Host Material ($g-C_3N_4$):** Melamine ($C_3H_6N_6$, $99.0\%$ purity) is chosen as the precursor because of its high condensation density and reliable yield ($\sim 35\%$).
*   **Co-catalyst ($NiS$):** Nickel Chloride Hexahydrate ($NiCl_2 \cdot 6H_2O$, $99.9\%$) and Thiourea ($CH_4N_2S$, $99.0\%$) are used.

### 2. Stoichiometric Mass Calculations
The calculations below determine the precise mass of precursor materials required to yield exactly $1.0\text{ g}$ of the final $NiS/g-C_3N_4$ composite. Molar masses: $M_{NiCl2\cdot 6H2O} = 237.69\text{ g/mol}$, $M_{Thiourea} = 76.12\text{ g/mol}$, $M_{NiS} = 90.753\text{ g/mol}$. 

*Note: For the hydrothermal deposition, a 2:1 molar ratio of Thiourea to Nickel is specified to ensure complete sulfidation and compensate for volatile sulfur losses.*

#### Recipe 1: Candidate 1 ($Ni_{0.006029}S_{0.006029}C_3N_4$ — $0.59\text{ wt\% } NiS$)
*   **Target $g-C_3N_4$ Host Mass:** $0.9941\text{ g}$ (requires **$2.840\text{ g}$** of Melamine precursor)
*   **Target $NiS$ Co-catalyst Mass:** $0.0059\text{ g}$ ($6.50\times 10^{-5}\text{ mol}$)
*   **Nickel Precursor ($NiCl_2 \cdot 6H_2O$):** **$15.45\text{ mg}$**
*   **Sulfur Precursor ($CH_4N_2S$):** **$4.95\text{ mg}$** (stoichiometric) / **$9.90\text{ mg}$** (recommended 2:1 molar excess)

#### Recipe 2: Candidate 2 ($Ni_{0.010271}S_{0.010271}C_3N_4$ — $1.00\text{ wt\% } NiS$)
*   **Target $g-C_3N_4$ Host Mass:** $0.9900\text{ g}$ (requires **$2.829\text{ g}$** of Melamine precursor)
*   **Target $NiS$ Co-catalyst Mass:** $0.0100\text{ g}$ ($1.10\times 10^{-4}\text{ mol}$)
*   **Nickel Precursor ($NiCl_2 \cdot 6H_2O$):** **$26.19\text{ mg}$**
*   **Sulfur Precursor ($CH_4N_2S$):** **$8.39\text{ mg}$** (stoichiometric) / **$16.78\text{ mg}$** (recommended 2:1 molar excess)

#### Recipe 3: Candidate 3 ($Ni_{0.008811}S_{0.008811}C_3N_4$ — $0.86\text{ wt\% } NiS$)
*   **Target $g-C_3N_4$ Host Mass:** $0.9914\text{ g}$ (requires **$2.833\text{ g}$** of Melamine precursor)
*   **Target $NiS$ Co-catalyst Mass:** $0.0086\text{ g}$ ($9.48\times 10^{-5}\text{ mol}$)
*   **Nickel Precursor ($NiCl_2 \cdot 6H_2O$):** **$22.52\text{ mg}$**
*   **Sulfur Precursor ($CH_4N_2S$):** **$7.21\text{ mg}$** (stoichiometric) / **$14.42\text{ mg}$** (recommended 2:1 molar excess)

### 3. Thermal Profiles and Synthesis Steps

#### Step 1: Synthesis of Graphitic Carbon Nitride ($g-C_3N_4$) Host
1.  Weigh the specified mass of Melamine precursor into a high-purity alumina crucible.
2.  Cover the crucible with a tight-fitting lid to minimize sublimation losses during calcination.
3.  Place the crucible into a muffle furnace and program the following thermal profile:
    *   **Heating Ramp:** $5^\circ\text{C/min}$ from room temperature to $550^\circ\text{C}$.
    *   **Calcination Dwell:** Hold at **$550^\circ\text{C}$ for 4 hours** in air.
    *   **Cooling Ramp:** Allow the furnace to cool naturally to room temperature.
4.  Retrieve the yellow product, grind it thoroughly using an agate mortar and pestle to obtain a fine powder, and store it in a desiccator.

#### Step 2: Hydrothermal Deposition of $NiS$ Co-catalyst
1.  Take the exact target mass of the synthesized $g-C_3N_4$ host and disperse it in $40\text{ mL}$ of deionized water.
2.  Subject the suspension to ultrasonication for **60 minutes** to guarantee complete and homogeneous dispersion of the $g-C_3N_4$ sheets.
3.  Add the specified masses of $NiCl_2 \cdot 6H_2O$ and Thiourea into the suspension. Stir vigorously for **30 minutes** to ensure complete dissolution and coordinate adsorption of the ions onto the carbon nitride surface.
4.  Transfer the homogeneous mixture to a $50\text{ mL}$ Teflon-lined stainless steel autoclave.
5.  Seal the autoclave and run the hydrothermal reaction:
    *   **Autoclave Ramp:** Heat to $180^\circ\text{C}$ (ramp rate $\approx 3^\circ\text{C/min}$).
    *   **Autoclave Dwell:** Maintain at **$180^\circ\text{C}$ for 12 hours** under autogenous pressure.
    *   **Cooling:** Allow the autoclave to cool naturally to room temperature.
6.  Collect the resulting composite powder via centrifugation. Wash the powder three times alternately with deionized water and absolute ethanol to remove any unreacted ions or free organic residues.
7.  Dry the final product in a vacuum oven at **$60^\circ\text{C}$ for 12 hours**.

---

## Phase 4: Photocatalytic Hydrogen Testing Protocol

To validate the physical catalysts in the laboratory, establish the following testing environment matching the dataset's baseline conditions.

### 1. Sacrificial Solution Composition
*   Prepare a **10 vol% Glycerol** aqueous solution (e.g., mix $10\text{ mL}$ of absolute glycerol with $90\text{ mL}$ of deionized water).
*   Add $50\text{ mg}$ of the synthesized composite catalyst powder to the solution. Sonicate briefly to ensure even dispersion before starting the test.

### 2. Reactor Setup and Light Source Configuration
*   **Reaction Vessel:** Use a double-walled Pyrex reaction flask connected to a temperature-controlled water bath set at **$25^\circ\text{C}$** to prevent thermal effects.
*   **Light Source:** Use a **$300\text{ W}$ Xenon lamp** adjusted to standard solar irradiance of **$100\text{ mW/cm}^2$** (1 sun equivalent).
*   **Spectral Filtering:** Equip the lamp with a cut-off filter ($\lambda > 420\text{ nm}$) to eliminate UV light. This isolates the visible-light photocatalytic performance, verifying the activity of the engineered host-guest composites.
*   **De-aeration:** Seal the reactor and bubble high-purity Argon (Ar) gas through the suspension for **30 minutes** before turning on the lamp to ensure strict anaerobic conditions.

### 3. Active Learning Closed-Loop Integration

```mermaid
graph LR
    A[Synthesize Physical Catalyst] --> B[Run Lab HER Testing]
    B --> C[Extract Experimental Rate]
    C --> D[Run add_experiment.py]
    D --> E[Update HTML Database]
    E --> F[Run data_clean.py]
    F --> G[Run pipeline_train.py]
    G --> H[Model Retrained & Saved]
    H --> A
```

Once experimental hydrogen evolution rates are measured in the laboratory, the results can be directly integrated back into the active learning loop.

1.  **Execute the update script** via the command-line interface. For example, to record the validation results of Candidate 1 yielding an experimental rate of $8650\ \mu\text{mol g}^{-1}\text{ h}^{-1}$:
    ```bash
    .venv\Scripts\python.exe add_experiment.py --catalyst "NiS/g-C3N4-Opt1" --composition "g-C3N4 + 0.59 wt% NiS" --bandgap "2.67" --vb "+1.58" --cb "-1.09" --hj "None" --cocat "NiS" --light "300 W Xe lamp (lambda > 420 nm)" --her "8650" --bet "50.5" --family "g-C3N4" --ref "Lab experiment, 2026"
    ```
2.  **Automated Execution Pipeline:**
    *   The script reads [add_experiment.py](file:///C:/Users/rauna/OneDrive/Desktop/Trial/add_experiment.py), parses the parameters, and appends the new record directly to the JavaScript dataset array inside [photocatalytic_H2_glycerol_catalyst_table.html](file:///C:/Users/rauna/OneDrive/Desktop/Trial/photocatalytic_H2_glycerol_catalyst_table.html).
    *   It immediately invokes [data_clean.py](file:///C:/Users/rauna/OneDrive/Desktop/Trial/data_clean.py) to parse the HTML table, perform cleaning/sanitization, and update the training CSV [catalyst_dataset_clean.csv](file:///C:/Users/rauna/OneDrive/Desktop/Trial/catalyst_dataset_clean.csv).
    *   It then runs [pipeline_train.py](file:///C:/Users/rauna/OneDrive/Desktop/Trial/pipeline_train.py), which performs 5-fold cross-validation and retrains the final model, saving the newly updated estimator to [matpipe_model.joblib](file:///C:/Users/rauna/OneDrive/Desktop/Trial/matpipe_model.joblib) so that subsequent screenings benefit from the newly added physical data point.
