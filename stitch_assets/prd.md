# Photocatalyst HER: ML-Guided Screening Dashboard
## Technical Specification & Architecture

### 1. Global Navigation & Structure
- **Sidebar/Navigation**: Mobile-optimized bottom bar or slide-out menu.
- **Micro-Cards**: Metrics grouped into "Validation Framework" and "Holdout Generalization".
- **Tooltips**: Contextual definitions for $LOMO-CV R^2$ and Conformal coverage.

### 2. Dataset Overview (`image_44739d.png`)
- High-level metrics: Total experiments (838), Semiconductors (21).
- Interactive tree-map for "Experiments by Semiconductor".
- HER distribution (log scale) with maximized canvas area.

### 3. Model Performance (`image_447415.png`)
- Reactive Canvas Charts for CV results with error bars.
- Toggle: Actual vs Predicted (Log Scale vs Original Scale).
- Residual Analysis Section ($Actual - Predicted$).

### 4. SHAP & Feature Analysis (`image_4473dc.jpg`)
- Interactive Beeswarm Plot area.
- Hover details for experimental vectors (catalyst loading, bandgap, power).
- Categorical filters: Catalyst Properties, Reaction Conditions, System Setup.

### 5. Virtual Screening (`image_4473bd.png`)
- Query-builder style filtering panel.
- Candidate table with expandable rows (e.g., `bivo4/rh`).
- "Why this candidate?" insights powered by model feature importance.

### 6. Publication Figures (`image_447454.png`)
- High-res asset gallery (300 DPI ready).
- One-tap download actions.