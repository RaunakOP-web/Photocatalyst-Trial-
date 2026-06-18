# Scientific Feature Audit

This report audits the current features in the photocatalyst HER prediction pipeline and recommends advanced descriptors from recent literature.

## Feature Classification

### A. Experimental Conditions
- **Current features**: `light_power_W`, `reaction_volume_mL`, `catalyst_loading_mg` (or `catalyst_loading_g_L`), `glycerol_concentration_v_pct`, `pH`, `light_type`, `light_source_type`, `sacrificial_donor`, `reactor_type`, `water_source`
- **Assessment**: Good coverage of basic setup, but lacks explicit photon flux (which depends on irradiance/distance) and catalyst-to-donor ratios.

### B. Semiconductor Properties
- **Current features**: `bandgap_eV`, `electron_affinity_eV`, `dielectric`, `crystal` (structural type code), `density`
- **Assessment**: Contains the core electronic structure properties. However, lacks explicit conduction band (CB) and valence band (VB) potentials relative to redox potentials (e.g., H+/H2 water reduction, glycerol oxidation), charge carrier effective masses, and exciton binding energies.

### C. Cocatalyst Properties
- **Current features**: `work_function`, `d_band_center`, `atomic_radius`, `electronegativity`, `price`
- **Assessment**: Captures key descriptor trends for metals. However, lacks explicit proxies for hydrogen adsorption free energy ($\Delta G_{H^*}$), surface energy, and overpotential.

### D. Structural Descriptors
- **Current features**: `form`, `structure` (e.g., anatase/rutile), `preparation_semiconductor`, `preparation_photocatalyst`
- **Assessment**: Captures synthesis history and crystal phase. Needs explicitly engineered flags for heterojunction status and co-catalyst synergy.

---

## Identified Missing Descriptors
1. **CB/VB Potentials relative to NHE/Vacuum**: Crucial for evaluating the thermodynamic driving force for both water/proton reduction and glycerol oxidation.
2. **Hydrogen Adsorption Free Energy ($\Delta G_{H^*}$)**: Sabatier principle descriptor for HER activity at the co-catalyst site.
3. **Effective Carrier Mass ($m_e^*$, $m_h^*$)**: Determines charge carrier mobility and recombination rates.
4. **Exciton Binding Energy Proxy**: Determines if photo-excited carriers easily separate or form stable, recombination-prone excitons.
5. **Surface Area Normalization**: HER should ideally scale with active surface area, not just weight.

## Leakage & Redundancy Risks
- **Redundancy**: `catalyst_loading_mg` and `catalyst_loading_g_L` represent the same physical constraint but in different units. We should standardize on concentration (`g_L`).
- **Leakage**: Any post-reaction performance indicators (like `AQY_pct` or `AQE_pct`) are already dropped during preprocessing, which is correct.
