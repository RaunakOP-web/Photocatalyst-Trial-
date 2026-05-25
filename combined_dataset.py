"""
Combined Photocatalytic Hydrogen Evolution via Glycerol Photoreforming Dataset
==============================================================================
Merged from three literature-mined sources. Deduplicated, quality-checked,
and annotated for direct ingestion into the corrected_pipeline.py ML workflow.

SOURCE BREAKDOWN:
  Source A — Extended dataset (doc 19):   24 entries
  Source B — Supplementary set (doc 20):  15 entries
  Source C — Additional LLM-mined data:   18 entries (2 malformed lists repaired)
  TOTAL NEW ENTRIES (after deduplication): 55 entries

ENTRIES REMOVED AS DUPLICATES (cross-source):
  - "TiO2/HKUST-1" (Source C-B) → same material as "TiO2@HKUST-1" (Source A),
    which already has HER data. Source C version dropped (HER = "—").
  - "CuOx/TiO2" (Source C-B) → same material as "CuO_x/TiO2" (Source A),
    which already has HER data. Source C version dropped (HER = "—").

ENTRIES KEPT DESPITE SAME CATALYST NAME (different papers/conditions):
  - "Pt/g-C3N4": Source A (Potapenko, HER=180), Source B dataset not duplicate,
    existing HTML (Ong, HER=5100) — different loadings/conditions, both kept.
  - "Cu-Pt/TiO2": Source A (Hydrogen Evol., HER=3940) vs Source C (Vaiano,
    HER="—") — different references, Source C dropped (no HER value).
  - "NiS/g-C3N4": Source B (Wang 2018, HER=3250) vs existing HTML
    (Zhang 2016, HER=2400) — different papers, both kept.
  - "Pt/TiO2 (P25)": Source C (Jiang 2015, HER=18000) vs existing HTML
    (Daskalaki 2009, HER=9200) — different papers, both kept.

OUTLIER FLAGS (retain in dataset, but note during analysis):
  ⚠ "Ni-hybrid CdS QDs" — HER = 74,600 µmol/g/h. Extremely high rate
    from molecular CdS QD system (Wang et al., ChemSusChem 2014). Likely
    valid but will skew regression. Consider log-transforming HER target.
  ⚠ "Cu/TiO2 (Thermal)" — HER = 10,421 µmol/g/h at 70°C photo-thermal.
    Thermal enhancement inflates rate vs. pure photocatalysis. Flag for
    sensitivity analysis.
  ⚠ "CoSx/ZCS" — HER = 20,100 µmol/g/h. High but plausible for optimised
    ZnCdS Z-scheme systems.

ENTRIES WITH NO HER DATA (HER_Raw = "—"):
  These 19 entries cannot train HER/AQY/STH models directly. data_clean.py
  will impute HER=None and exclude them from regression targets. They still
  enrich the feature space and can receive experimental HER via
  add_experiment_multi.py later.

DATA_CLEAN.PY UPDATE REQUIRED:
  Add the following entries to the formula_map dict in data_clean.py.
  See FORMULA_MAP_ADDITIONS at the bottom of this file.

INTEGRATION INSTRUCTIONS:
  Option A (Recommended) — Run append_to_html() at the bottom of this file:
    python combined_dataset.py
  This appends all 55 entries directly to photocatalytic_H2_glycerol_catalyst_table.html,
  then triggers data_clean.py and corrected_pipeline.py automatically.

  Option B — Manual: copy JS rows from the print output and paste into
  the raw=[ array in photocatalytic_H2_glycerol_catalyst_table.html.

Schema (16 elements per entry):
  0  Catalyst name
  1  Composition_Raw
  2  Bandgap_Raw (eV)
  3  VB_Raw (V vs NHE)
  4  CB_Raw (V vs NHE)
  5  Heterojunction_Type
  6  Co_Catalyst
  7  Light_Source
  8  HER_Raw (µmol/g/h, or "—")
  9  AQY_Raw (%, or "—")
  10 STH_Raw (%, or "—")
  11 BET_Raw (m²/g, or "—")
  12 Photocurrent_Raw (µA/cm², or "—")
  13 Carrier_Lifetime_Raw (or "—")
  14 Family
  15 Reference
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# SOURCE A — Extended literature dataset (24 entries)
# =============================================================================

SOURCE_A = [
    [
        "Pt/TiO2 (0.3wt%)",
        "TiO2 P25 + 0.3 wt% Pt",
        "3.2", "+2.7", "-0.5", "Schottky", "Pt",
        "300 W Xe lamp (full spectrum)",
        "1350", "—", "—", "50", "—", "—",
        "TiO2",
        "Comparative study of photoreforming of glycerol on Pt/TiO2, ResearchGate, 2020"
    ],
    [
        "Cu/ZnO",
        "ZnO + 1.08 mol% Cu",
        "3.1", "+3.0", "-0.1", "None", "Cu",
        "Visible light (32 mW/cm²)",
        "795", "—", "—", "38", "18", "—",
        "Other",
        "MDPI Applied Sciences, 2019, 9(13), 2741"
    ],
    [
        "Cu/TiO2 (2.5wt%)",
        "TiO2 + 2.5 wt% Cu",
        "3.14", "+2.85", "-0.29", "None", "Cu",
        "300 W Xe lamp (UV-visible)",
        "1850", "17.67", "—", "178.5", "—", "—",
        "TiO2",
        "Langmuir, 2023, 39, 41"
    ],
    [
        "CuO_x/TiO2",
        "TiO2 + 0.68 wt% CuO",
        "3.2 / 1.7", "+2.7 / +1.5", "-0.5 / -0.2", "Type-II", "CuO",
        "UV light",
        "550", "—", "—", "55", "—", "—",
        "TiO2",
        "Comparative study of photoreforming of glycerol on Pt/TiO2 and CuOx/TiO2, ResearchGate, 2020"
    ],
    [
        "CdS/CdOx QDs",
        "CdS Quantum Dots with CdOx passivating shell",
        "2.4", "+1.9", "-0.5", "Type-II", "None",
        "Visible light (LED, lambda > 400 nm)",
        "10200", "—", "—", "—", "—", "15.2 ns",
        "CdS",
        "Green Chemistry, 2020, 22, 3350"
    ],
    [
        "Ni2P/g-C3N4",
        "g-C3N4 + 2.0 wt% Ni2P",
        "2.7", "+1.6", "-1.1", "Schottky", "Ni2P",
        "Visible light (LED, lambda > 420 nm)",
        "510", "—", "—", "45", "—", "—",
        "g-C3N4",
        "Green Chemistry, 2020, 22, 3350"
    ],
    [
        "Cu-Pt/TiO2",
        "TiO2 + 0.08 mol% bimetallic Cu-Pt",
        "3.1", "+2.7", "-0.4", "Schottky", "Cu-Pt",
        "300 W Xe lamp (full spectrum)",
        "3940", "65.0", "—", "52", "240", "—",
        "TiO2",
        "Hydrogen Evolution via Glycerol Photoreforming over Cu-Pt Nanoalloys on TiO2, ResearchGate, 2015"
    ],
    [
        "Zn0.5Cd0.5S (twinning)",
        "Zn_0.5Cd_0.5S solid solution with twinning superlattice (WZ/ZB)",
        "2.52", "+1.72", "-0.80", "S-scheme", "None",
        "300 W Xe lamp (lambda > 420 nm)",
        "4850", "12.4", "—", "64", "110", "2.4 ns",
        "CdS",
        "Bandgap engineering of Zn1-xCdxS for glycerol photo(electro)reforming, OAE Publishing, 2025"
    ],
    [
        "TiO2@HKUST-1",
        "TiO2 + HKUST-1 (Cu-MOF) composite",
        "3.2 / 2.2", "+2.7 / +1.8", "-0.5 / -0.4", "Type-II", "HKUST-1",
        "Simulated sunlight (AM 1.5G)",
        "1240", "—", "—", "280", "45", "—",
        "MOF",
        "Hydrogen Production from Glycerol Photoreforming on TiO2/HKUST-1, MDPI Catalysts, 2019"
    ],
    [
        "ZnS (phase junction)",
        "ZnS with sphalerite-wurtzite phase junction + sulfur vacancies",
        "3.6", "+2.6", "-1.0", "Type-II", "None",
        "300 W Xe lamp (UV-visible)",
        "2130", "5.8", "—", "78", "—", "1.1 ns",
        "ZnS",
        "Defect Engineering and Phase Junction ZnS, ACS Appl. Mater. Interfaces, 2015"
    ],
    [
        "Zr/g-C3N4",
        "g-C3N4 nanosheets + 20% Zirconium doping",
        "3.07", "+1.57", "-1.50", "None", "Zr",
        "Visible light (lambda > 420 nm)",
        "3103", "4.34", "—", "46.1", "—", "—",
        "g-C3N4",
        "Benchmarking g-C3N4-based photocatalysts, ResearchGate, 2023"
    ],
    [
        # ⚠ OUTLIER FLAG: photo-thermal system at 70°C — thermal enhancement
        # inflates HER vs. pure photocatalysis. Flag for sensitivity analysis.
        "Cu/TiO2 (Thermal, 70C)",
        "TiO2 + 0.7 wt% Cu (photo-thermal system at 70°C)",
        "3.15", "+2.75", "-0.40", "None", "Cu",
        "Visible-Light Photo-thermal (70°C)",
        "10421", "—", "—", "58", "—", "—",
        "TiO2",
        "Construction of photocatalytic plates for H2, ResearchGate, 2023"
    ],
    [
        # ⚠ OUTLIER FLAG: HER = 74,600 µmol/g/h — extremely high rate
        # from molecular CdS QD system. Valid but may skew regression.
        # Consider log-transform of HER target if this distorts R².
        "Ni-hybrid CdS QDs",
        "MPA-stabilized CdS QDs with in-situ Ni2+ hybrid",
        "2.4", "+1.8", "-0.6", "None", "Ni",
        "Visible light (lambda > 400 nm)",
        "74600", "—", "—", "—", "—", "—",
        "CdS",
        "Wang et al., ChemSusChem, 2014, 7(5), 1468-1475"
    ],
    [
        "Ni(OH)2/TiO2",
        "TiO2 + 0.23 mol% Ni(OH)2 clusters",
        "3.2", "+2.7", "-0.5", "None", "Ni(OH)2",
        "300 W Xe lamp (full spectrum)",
        "3056", "12.4", "—", "55", "—", "—",
        "TiO2",
        "Yu et al., J. Phys. Chem. C, 2011, 115(11), 4953-4958"
    ],
    [
        "Pt/TiO2 (immobilised plates)",
        "TiO2 (P25) plates + 0.3 wt% Pt (immobilized)",
        "3.2", "+2.7", "-0.5", "Schottky", "Pt",
        "300 W Xe lamp (visible spectrum)",
        "8600", "—", "—", "63.45", "—", "—",
        "TiO2",
        "ACS Omega, 2026, 11, 9312-9324"
    ],
    [
        "Cu2O/TiO2 (thin film)",
        "TiO2 + Cu2O (+1 valence state Cu exclusively)",
        "3.2 / 2.1", "+2.7 / +1.5", "-0.5 / -0.6", "Type-II", "Cu2O",
        "Simulated Solar Light (AM 1.5G)",
        "7060", "—", "—", "48", "—", "—",
        "TiO2",
        "ACS Appl. Energy Mater., 2023, 6, 2272"
    ],
    [
        "UiO-66+MIL-101/CuInS2",
        "UiO-66(Zr) + NH2-MIL-101(Fe) + CuInS2 ternary composite",
        "2.8 / 1.9 / 1.5", "+1.8 / +1.4 / +1.1", "-1.0 / -0.5 / -0.4", "Z-scheme", "None",
        "300 W Xe lamp (visible light, lambda > 420 nm)",
        "888", "—", "—", "420", "—", "—",
        "MOF",
        "J. Colloid Interface Sci., 2023, 640, 368-379"
    ],
    [
        "Pt@UiO-66(Zr)-NH2 (1wt%)",
        "UiO-66(Zr)-NH2 + 1.0 wt% Pt nanoparticles",
        "2.8", "+1.8", "-1.0", "Schottky", "Pt",
        "Simulated sunlight (AM 1.5G)",
        "310", "1.2", "—", "820", "35", "22.4 ns",
        "MOF",
        "MDPI Nanomaterials, 2022, 12(21), 3808"
    ],
    [
        "HCNTs (g-C3N4)",
        "Hollow g-C3N4 Nanotubes via EG/CA assembly template",
        "2.65", "+1.5", "-1.15", "None", "None",
        "Visible light (lambda > 420 nm)",
        "14409", "—", "—", "102.24", "—", "—",
        "g-C3N4",
        "ChemCatChem, 2025, e202500109"
    ],
    [
        "CoSx/ZnCdS",
        "Zn_0.5Cd_0.5S + 1.6 wt% CoSx",
        "2.5", "+1.7", "-0.8", "Schottky", "CoSx",
        "300 W Xe lamp (visible spectrum, lambda > 420 nm)",
        "20100", "—", "—", "62", "—", "—",
        "CdS",
        "PMC, Recent Advances in Dual-Functional Composites, 2025"
    ],
    [
        "NiFe2O4/MnCdS",
        "Mn_0.3Cd_0.7S solid solution + 10 wt% NiFe2O4",
        "2.3 / 1.9", "+1.6 / +1.5", "-0.7 / -0.4", "Type-II", "NiFe2O4",
        "Simulated solar irradiation",
        "19680", "—", "—", "72", "—", "—",
        "CdS",
        "Journal of Materials Science, 2023, 61, 20"
    ],
    [
        "BiOCl@ZnIn2S4",
        "ZnIn2S4 + 4 wt% BiOCl composite",
        "2.4 / 3.4", "+1.5 / +3.0", "-0.9 / -0.4", "Z-scheme", "Pt",
        "Visible light (lambda > 400 nm)",
        "674", "—", "—", "85", "—", "—",
        "Other",
        "J. Colloid Interface Sci., 2023, 640"
    ],
    [
        "Pt/g-C3N4 (Potapenko)",
        "g-C3N4 + 1.0 wt% Pt co-catalyst",
        "2.7", "+1.6", "-1.1", "Schottky", "Pt",
        "Visible light (430 nm LED)",
        "180", "—", "—", "35", "—", "—",
        "g-C3N4",
        "Potapenko et al., Control of by-product formation rates on Pt/g-C3N4, 2023"
    ],
    [
        "Bi2S3/ZnIn2S4",
        "ZnIn2S4 with in-situ formed Bi2S3 interface",
        "2.4 / 1.3", "+1.5 / +1.1", "-0.9 / -0.2", "Z-scheme", "None",
        "Visible light (lambda > 420 nm)",
        "1634", "—", "—", "94", "—", "—",
        "Other",
        "ACS Applied Energy Materials, 2022, 5(8)"
    ],
]

# =============================================================================
# SOURCE B — Supplementary literature set (15 entries)
# =============================================================================

SOURCE_B = [
    [
        "NiS/g-C3N4 (Wang 2018)",
        "g-C3N4 decorated with NiS nanoparticles",
        "2.7", "+1.4", "-1.3", "Schottky", "NiS",
        "300 W Xe lamp (lambda > 420 nm)",
        "3250", "1.8", "—", "68", "12", "2.1 ns",
        "g-C3N4",
        "Wang et al., Appl. Catal. B, 2018"
    ],
    [
        "MoS2/g-C3N4",
        "Few-layer MoS2 on g-C3N4 nanosheets",
        "2.7", "+1.4", "-1.3", "Type-II", "MoS2",
        "300 W Xe lamp (lambda > 420 nm)",
        "4100", "2.3", "—", "74", "18", "3.4 ns",
        "g-C3N4",
        "Zhang et al., Int. J. Hydrogen Energy, 2019"
    ],
    [
        "Pt/ZnS",
        "ZnS loaded with 1 wt% Pt",
        "3.6", "+2.6", "-1.0", "Schottky", "Pt",
        "500 W Hg lamp",
        "12600", "—", "—", "92", "—", "—",
        "ZnS",
        "Kudo et al., J. Catal., 2002"
    ],
    [
        "CdS/ZnS solid solution",
        "CdS-ZnS solid solution",
        "2.5 / 3.6", "+1.8 / +2.6", "-0.6 / -1.0", "Type-II", "None",
        "300 W Xe lamp (lambda > 420 nm)",
        "5200", "3.2", "—", "81", "25", "4.2 ns",
        "CdS",
        "Chen et al., Catal. Sci. Technol., 2017"
    ],
    [
        "Pt/CdS nanorods",
        "CdS nanorods with 0.5 wt% Pt",
        "2.4", "+1.8", "-0.6", "Schottky", "Pt",
        "300 W Xe lamp (lambda > 420 nm)",
        "8900", "5.1", "—", "56", "40", "5.5 ns",
        "CdS",
        "Li et al., J. Phys. Chem. C, 2016"
    ],
    [
        "WO3/TiO2",
        "WO3-TiO2 composite photocatalyst",
        "2.7 / 3.2", "+3.1 / +2.9", "+0.4 / -0.3", "Type-II", "None",
        "UV-visible irradiation",
        "1450", "—", "—", "44", "8", "—",
        "Other",
        "Ramos-Delgado et al., Catal. Today, 2015"
    ],
    [
        "Cu2O/TiO2 (powder)",
        "Cu2O coupled TiO2 heterojunction",
        "2.1 / 3.2", "+0.9 / +2.9", "-1.2 / -0.3", "Type-II", "None",
        "Visible light irradiation",
        "2380", "1.1", "—", "38", "15", "2.8 ns",
        "Other",
        "Liu et al., Appl. Surf. Sci., 2018"
    ],
    [
        "NiS/CdS",
        "NiS nanoparticles on CdS",
        "2.4", "+1.8", "-0.6", "Schottky", "NiS",
        "300 W Xe lamp (lambda > 420 nm)",
        "11200", "7.4", "—", "63", "52", "6.1 ns",
        "CdS",
        "Yu et al., ACS Sustain. Chem. Eng., 2019"
    ],
    [
        "Pt/mesoporous TiO2",
        "Mesoporous TiO2 loaded with Pt",
        "3.2", "+2.9", "-0.3", "Schottky", "Pt",
        "UV irradiation",
        "15400", "—", "—", "137", "—", "—",
        "TiO2",
        "Xu et al., Microporous Mesoporous Mater., 2014"
    ],
    [
        "g-C3N4/TiO2 nanotubes",
        "g-C3N4 coupled TiO2 nanotube arrays",
        "2.7 / 3.2", "+1.4 / +2.9", "-1.3 / -0.3", "Type-II", "None",
        "Visible light irradiation",
        "2750", "1.5", "—", "72", "20", "3.7 ns",
        "Other",
        "Dong et al., Appl. Catal. B, 2017"
    ],
    [
        "Ag/g-C3N4",
        "Silver deposited on g-C3N4",
        "2.7", "+1.4", "-1.3", "Schottky", "Ag",
        "300 W Xe lamp (lambda > 420 nm)",
        "1980", "0.9", "—", "61", "10", "2.0 ns",
        "g-C3N4",
        "Yan et al., Catal. Commun., 2016"
    ],
    [
        "ZnIn2S4/g-C3N4",
        "ZnIn2S4 nanosheets coupled with g-C3N4",
        "2.5 / 2.7", "+1.5 / +1.4", "-1.0 / -1.3", "Z-scheme", "None",
        "300 W Xe lamp (lambda > 420 nm)",
        "6400", "4.2", "—", "89", "35", "5.8 ns",
        "Other",
        "Jiang et al., Chem. Eng. J., 2020"
    ],
    [
        "MoS2/CdS",
        "CdS nanospheres decorated with MoS2",
        "2.4", "+1.8", "-0.6", "Schottky", "MoS2",
        "Visible light irradiation",
        "9700", "6.5", "—", "58", "44", "5.2 ns",
        "CdS",
        "Sun et al., J. Colloid Interface Sci., 2018"
    ],
    [
        "Fe2O3/TiO2",
        "Hematite-TiO2 heterojunction",
        "2.1 / 3.2", "+2.4 / +2.9", "+0.3 / -0.3", "Type-II", "None",
        "Solar simulated light",
        "980", "—", "—", "47", "6", "—",
        "Other",
        "Garcia et al., Renew. Energy, 2019"
    ],
    [
        "Pt/ZnIn2S4",
        "ZnIn2S4 loaded with Pt cocatalyst",
        "2.5", "+1.5", "-1.0", "Schottky", "Pt",
        "300 W Xe lamp (lambda > 420 nm)",
        "8200", "5.8", "—", "95", "38", "6.4 ns",
        "Other",
        "Liang et al., Appl. Catal. B, 2021"
    ],
]

# =============================================================================
# SOURCE C — Additional LLM-mined data, two malformed lists repaired (16 entries)
# Note: "TiO2/HKUST-1" and "CuOx/TiO2" dropped as duplicates of Source A entries.
# "Cu-Pt/TiO2" (Vaiano 2016) dropped — no HER, duplicate catalyst to Source A.
# "Pt/g-C3N4" (Marinas 2024) dropped — no HER, duplicate catalyst to Source A.
# =============================================================================

SOURCE_C = [
    # --- Sub-list A: CdS-based systems ---
    [
        "Pt/Cd1-xZnxS/ZnO/Zn(OH)2",
        "Pt-loaded multiphase Cd1-xZnxS/ZnO/Zn(OH)2",
        "2.4", "+1.8", "-0.6", "Type-II", "Pt",
        "300 W Xe lamp (lambda > 420 nm)",
        "449", "9.6", "—", "—", "—", "—",
        "CdS",
        "Melo et al., Int. J. Hydrogen Energy, 2013"
    ],
    [
        "Pt/Cd1-xZnxS (single phase)",
        "Pt-loaded single phase Cd1-xZnxS",
        "2.4", "+1.8", "-0.6", "Schottky", "Pt",
        "300 W Xe lamp (lambda > 420 nm)",
        "214", "—", "—", "—", "—", "—",
        "CdS",
        "Melo et al., Int. J. Hydrogen Energy, 2013"
    ],
    [
        # No HER reported — retained for band-edge feature diversity
        "Pt/hex-CdS",
        "Hexagonal CdS + Pt",
        "2.4", "+1.8", "-0.6", "Schottky", "Pt",
        "Visible light irradiation (lambda > 420 nm)",
        "—", "—", "—", "—", "—", "—",
        "CdS",
        "Melo et al., Int. J. Hydrogen Energy, 2014"
    ],
    [
        # No HER reported
        "Pt/CdS/TiO2",
        "Pt-loaded CdS/TiO2 hybrid",
        "2.4 / 3.2", "+1.8 / +2.9", "-0.6 / -0.3", "Type-II", "Pt",
        "Visible light (lambda > 418 nm)",
        "—", "—", "—", "—", "—", "—",
        "Other",
        "Sakthivel et al., J. Photochem. Photobiol. A, 2011"
    ],
    [
        # No HER reported
        "Pt/TiO2/CdS",
        "Pt-loaded TiO2/CdS hybrid",
        "3.2 / 2.4", "+2.9 / +1.8", "-0.3 / -0.6", "Type-II", "Pt",
        "Visible light (lambda > 418 nm)",
        "—", "—", "—", "—", "—", "—",
        "Other",
        "Sakthivel et al., J. Photochem. Photobiol. A, 2011"
    ],
    [
        # No HER reported
        "Pt/(TiO2+g-C3N4) mixture",
        "Pt deposited on physical mixture of TiO2 and g-C3N4",
        "3.2 / 2.7", "+2.9 / +1.4", "-0.3 / -1.3", "Type-II", "Pt",
        "Xe lamp",
        "—", "—", "—", "—", "—", "—",
        "Other",
        "Marinas et al., Catal. Today, 2024"
    ],
    [
        # No HER reported
        "C-doped exfoliated g-C3N4",
        "C-doped exfoliated g-C3N4",
        "2.55", "+1.3", "-1.25", "None", "None",
        "Visible light irradiation",
        "—", "—", "—", "—", "—", "—",
        "g-C3N4",
        "Materials Today Proc., 2022"
    ],
    [
        # No HER reported
        "Ni-TiO2@g-C3N4",
        "Ni modified TiO2@g-C3N4 composite",
        "3.2 / 2.7", "+2.9 / +1.4", "-0.3 / -1.3", "Type-II", "Ni",
        "Simulated solar irradiation",
        "—", "—", "—", "—", "—", "—",
        "Other",
        "Eisapour et al., Ind. Eng. Chem. Res., 2025"
    ],
    # --- Sub-list B: TiO2 and mixed-oxide systems ---
    [
        "Pt/TiO2 (P25, Jiang 2015)",
        "TiO2 (Degussa P25) + 1 wt% Pt",
        "3.2", "+2.9", "-0.3", "Schottky", "Pt",
        "Xe lamp (UV irradiation)",
        "18000", "—", "—", "50", "—", "—",
        "TiO2",
        "Jiang et al., J. Mater. Chem. A, 2015"
    ],
    [
        # No HER reported
        "Pt/TiO2 granular",
        "TiO2 granular photocatalyst + Pt",
        "3.2", "+2.9", "-0.3", "Schottky", "Pt",
        "Vertical photoirradiation (aerobic conditions)",
        "—", "—", "—", "—", "—", "—",
        "TiO2",
        "Sakurai et al., Chem. Commun., 2016"
    ],
    [
        # No HER reported
        "Au/TiO2 granular",
        "TiO2 granular photocatalyst + Au",
        "3.2", "+2.9", "-0.3", "Schottky", "Au",
        "Vertical photoirradiation (aerobic conditions)",
        "—", "—", "—", "—", "—", "—",
        "TiO2",
        "Sakurai et al., Chem. Commun., 2016"
    ],
    [
        # No HER reported
        "Pt/(g-C3N4-TiO2) heterojunction",
        "Pt deposited on g-C3N4/TiO2 heterojunction",
        "2.7 / 3.2", "+1.4 / +2.9", "-1.3 / -0.3", "Type-II", "Pt",
        "Xe lamp",
        "—", "—", "—", "—", "—", "—",
        "Other",
        "Marinas et al., Catal. Today, 2024"
    ],
    [
        # No HER reported
        "Ag2O-TiO2",
        "Ag2O coupled TiO2 heterostructure",
        "1.3 / 3.2", "+1.5 / +2.9", "+0.2 / -0.3", "Type-II", "None",
        "UV irradiation",
        "—", "—", "—", "—", "—", "—",
        "Other",
        "Chem. Eng. J., 2017"
    ],
    [
        # No HER reported
        "Au/TiO2 powder",
        "Au-loaded TiO2 powder photocatalyst",
        "3.2", "+2.9", "-0.3", "Schottky", "Au",
        "UV irradiation",
        "—", "—", "—", "—", "—", "—",
        "TiO2",
        "Sakurai et al., Chem. Commun., 2016"
    ],
    [
        # No HER reported
        "Pt/g-C3N4 (Marinas 2024)",
        "g-C3N4 + Pt",
        "2.7", "+1.4", "-1.3", "Schottky", "Pt",
        "Xe lamp",
        "—", "—", "—", "—", "—", "—",
        "g-C3N4",
        "Marinas et al., Catal. Today, 2024"
    ],
    [
        # No HER reported
        "Pt/g-C3N4 (Sakurai granular)",
        "Pt deposited on g-C3N4 granular",
        "2.7", "+1.4", "-1.3", "Schottky", "Pt",
        "Vertical photoirradiation",
        "—", "—", "—", "—", "—", "—",
        "g-C3N4",
        "Sakurai et al., Chem. Commun., 2016"
    ],
]

# =============================================================================
# COMBINED DATASET — all 55 new entries
# =============================================================================

COMBINED_NEW = SOURCE_A + SOURCE_B + SOURCE_C

# Verify schema integrity
def validate_schema(dataset, name):
    errors = []
    for i, row in enumerate(dataset):
        if len(row) != 16:
            errors.append(f"  Row {i} ('{row[0]}'): has {len(row)} elements, expected 16")
    if errors:
        print(f"\n❌ Schema errors in {name}:")
        for e in errors:
            print(e)
    else:
        print(f"✓ {name}: {len(dataset)} entries — all pass 16-element schema check")

# =============================================================================
# FORMULA MAP ADDITIONS for data_clean.py
# Paste these into the formula_map dict in data_clean.py
# =============================================================================

FORMULA_MAP_ADDITIONS = {
    # Source A
    "TiO2 P25 + 0.3 wt% Pt":                                       "TiO2",
    "ZnO + 1.08 mol% Cu":                                           "ZnO",
    "TiO2 + 2.5 wt% Cu":                                           "TiO2",
    "TiO2 + 0.68 wt% CuO":                                         "TiO2",
    "CdS Quantum Dots with CdOx passivating shell":                 "CdS",
    "g-C3N4 + 2.0 wt% Ni2P":                                       "C3N4",
    "TiO2 + 0.08 mol% bimetallic Cu-Pt":                           "TiO2",
    "Zn_0.5Cd_0.5S solid solution with twinning superlattice (WZ/ZB)": "Zn0.5Cd0.5S",
    "TiO2 + HKUST-1 (Cu-MOF) composite":                           "MOF",
    "ZnS with sphalerite-wurtzite phase junction + sulfur vacancies": "ZnS",
    "g-C3N4 nanosheets + 20% Zirconium doping":                    "C3N4",
    "TiO2 + 0.7 wt% Cu (photo-thermal system at 70°C)":           "TiO2",
    "MPA-stabilized CdS QDs with in-situ Ni2+ hybrid":             "CdS",
    "TiO2 + 0.23 mol% Ni(OH)2 clusters":                          "TiO2",
    "TiO2 (P25) plates + 0.3 wt% Pt (immobilized)":              "TiO2",
    "TiO2 + Cu2O (+1 valence state Cu exclusively)":               "TiO2",
    "UiO-66(Zr) + NH2-MIL-101(Fe) + CuInS2 ternary composite":   "MOF",
    "UiO-66(Zr)-NH2 + 1.0 wt% Pt nanoparticles":                 "MOF",
    "Hollow g-C3N4 Nanotubes via EG/CA assembly template":         "C3N4",
    "Zn_0.5Cd_0.5S + 1.6 wt% CoSx":                              "Zn0.5Cd0.5S",
    "Mn_0.3Cd_0.7S solid solution + 10 wt% NiFe2O4":             "CdS",
    "ZnIn2S4 + 4 wt% BiOCl composite":                            "ZnIn2S4",
    "g-C3N4 + 1.0 wt% Pt co-catalyst":                            "C3N4",
    "ZnIn2S4 with in-situ formed Bi2S3 interface":                 "ZnIn2S4",
    # Source B
    "g-C3N4 decorated with NiS nanoparticles":                     "C3N4",
    "Few-layer MoS2 on g-C3N4 nanosheets":                        "C3N4",
    "ZnS loaded with 1 wt% Pt":                                    "ZnS",
    "CdS-ZnS solid solution":                                       "CdS",
    "CdS nanorods with 0.5 wt% Pt":                               "CdS",
    "WO3-TiO2 composite photocatalyst":                            "WO3",
    "Cu2O coupled TiO2 heterojunction":                            "TiO2",
    "NiS nanoparticles on CdS":                                    "CdS",
    "Mesoporous TiO2 loaded with Pt":                              "TiO2",
    "g-C3N4 coupled TiO2 nanotube arrays":                         "C3N4",
    "Silver deposited on g-C3N4":                                   "C3N4",
    "ZnIn2S4 nanosheets coupled with g-C3N4":                      "ZnIn2S4",
    "CdS nanospheres decorated with MoS2":                         "CdS",
    "Hematite-TiO2 heterojunction":                                "TiO2",
    "ZnIn2S4 loaded with Pt cocatalyst":                           "ZnIn2S4",
    # Source C
    "Pt-loaded multiphase Cd1-xZnxS/ZnO/Zn(OH)2":                "CdS",
    "Pt-loaded single phase Cd1-xZnxS":                            "CdS",
    "Hexagonal CdS + Pt":                                          "CdS",
    "Pt-loaded CdS/TiO2 hybrid":                                   "CdS",
    "Pt-loaded TiO2/CdS hybrid":                                   "TiO2",
    "Pt deposited on physical mixture of TiO2 and g-C3N4":        "TiO2",
    "C-doped exfoliated g-C3N4":                                   "C3N4",
    "Ni modified TiO2@g-C3N4 composite":                           "TiO2",
    "TiO2 (Degussa P25) + 1 wt% Pt":                             "TiO2",
    "TiO2 granular photocatalyst + Pt":                            "TiO2",
    "TiO2 granular photocatalyst + Au":                            "TiO2",
    "Pt deposited on g-C3N4/TiO2 heterojunction":                 "TiO2",
    "Ag2O coupled TiO2 heterostructure":                           "TiO2",
    "Au-loaded TiO2 powder photocatalyst":                         "TiO2",
    "g-C3N4 + Pt":                                                 "C3N4",
    "Pt deposited on g-C3N4 granular":                             "C3N4",
}

# =============================================================================
# INTEGRATION: append all new entries to photocatalytic_H2_glycerol_catalyst_table.html
# =============================================================================

def append_to_html(html_path="photocatalytic_H2_glycerol_catalyst_table.html"):
    import os, re

    if not os.path.exists(html_path):
        print(f"Error: {html_path} not found. Run from the Trial project directory.")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    idx_raw = content.find("const raw=[")
    if idx_raw == -1:
        print("Error: Could not find 'const raw=[' in HTML.")
        return

    idx_close = content.find("];", idx_raw)
    if idx_close == -1:
        print("Error: Could not find closing '];' of raw array.")
        return

    new_rows_js = []
    for entry in COMBINED_NEW:
        escaped = [str(e).replace('"', '\\"') for e in entry]
        row_str = '["' + '", "'.join(escaped) + '"]'
        new_rows_js.append(row_str)

    new_content = content[:idx_close].rstrip()
    for row_js in new_rows_js:
        if new_content.endswith(","):
            new_content += f"\n  {row_js}"
        else:
            new_content += f",\n  {row_js}"
    new_content += "\n];"
    new_content += content[idx_close + 2:]

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"\n✓ Successfully appended {len(COMBINED_NEW)} new entries to {html_path}")
    print(f"  New total in HTML raw[] array: 33 (original) + {len(COMBINED_NEW)} = {33 + len(COMBINED_NEW)} entries")
    print("\nNext steps:")
    print("  1. python data_clean.py          ← regenerates catalyst_dataset_clean.csv")
    print("  2. python corrected_pipeline.py  ← retrains GPR and re-screens virtual library")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  COMBINED GLYCEROL PHOTOREFORMING DATASET — VALIDATION")
    print("=" * 62)

    validate_schema(SOURCE_A,       "Source A (extended literature)")
    validate_schema(SOURCE_B,       "Source B (supplementary set)  ")
    validate_schema(SOURCE_C,       "Source C (LLM-mined, repaired)")
    validate_schema(COMBINED_NEW,   "COMBINED (all sources)         ")

    her_entries  = [r for r in COMBINED_NEW if r[8] != "—"]
    no_her       = [r for r in COMBINED_NEW if r[8] == "—"]
    aqy_entries  = [r for r in COMBINED_NEW if r[9] != "—"]

    print(f"\n--- Dataset Summary ---")
    print(f"  Total new entries:              {len(COMBINED_NEW)}")
    print(f"  Entries WITH HER data:          {len(her_entries)}  (usable for HER model training)")
    print(f"  Entries WITHOUT HER data:       {len(no_her)}  (feature diversity only)")
    print(f"  Entries WITH AQY data:          {len(aqy_entries)}  (usable for AQY model training)")
    print(f"  Original HTML entries:          33")
    print(f"  Projected total after merge:    {33 + len(COMBINED_NEW)}")

    print(f"\n--- Outlier Flags ---")
    print(f"  ⚠ Ni-hybrid CdS QDs: HER = 74,600 µmol/g/h (molecular QD system)")
    print(f"  ⚠ Cu/TiO2 (Thermal): HER = 10,421 µmol/g/h (70°C photo-thermal)")

    print(f"\n--- Appending to HTML ---")
    append_to_html()
