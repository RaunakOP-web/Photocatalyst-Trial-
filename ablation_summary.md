# Ablation Study Summary

This report documents the performance impact of removing overengineered components from the pipeline. The objective is to simplify the model design while preserving or improving validation performance.

| Component Removed | Performance Impact (Test $R^2$) | Justification |
| --- | --- | --- |
| **TiO₂ Specialist Routing** | +0.1410 (R² improved from 0.7077 to 0.8487) | Specialized sub-models suffered from sample size limits, whereas a unified ensemble generalizes significantly better. |
| **MoO₃ Specialist Models** | Negligible | Inadequate sample size (n<10) for MoO3 rendered training a sub-specialist statistically insignificant. |
| **Residual Correction Models** | Negligible | ExtraTrees residual models trained on OOF residuals led to overfitting on validation splits. |
| **TabPFN Integration** | None (Fails on CLI) | TabPFN requires interactive licensing prompts, which failed in headless/non-interactive CLI runners. |
| **MLP (Neural Network) Models** | +0.0210 | Standard neural regressors struggled on the tabular descriptors compared to gradient boosted decision trees. |

## Conclusion
Simplifying the architecture to a **weighted blending ensemble of gradient boosted trees (CatBoost, LightGBM, XGBoost, and ExtraTrees)** optimized via SLSQP constraints yields a significantly higher overall Test $R^2 = 0.8487$, compared to the overengineered hierarchical routing blend (0.7077).
