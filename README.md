# The Phi Signature in Human Cognition

**Embedding Geometry Predicts Insight Behavior in the Compound Remote Associates Test**

Borja Azpiroz, 2026

Companion paper: *The Phi Constant of Knowledge: Cross-Domain Geometric Invariance in Semantic Embedding Spaces* (Azpiroz, 2026)

---

## Key Result

Three independent measurements converge within delta < 0.004 of 1/phi^2 = 0.382:

| Measurement | Value | Source |
|---|---|---|
| 1/phi^2 (theory) | 0.38197 | Mathematics |
| BGE-base centroid (CRA triads) | 0.38384 | This paper |
| Youden threshold | 0.38600 | Companion paper |

Bootstrap 95% CI [0.374, 0.394] contains both 1/phi^2 and the Youden threshold.

## Results Summary

| Test | Result |
|---|---|
| **Triple convergence** | delta < 0.004 (CI confirmed) |
| **Cascade** | d(Internal) < d(Noise) in 11/11 models, p = 0.00014 |
| **BGE-base correlation** | r = -0.282, permutation p = 0.020 |
| **Meta-analysis** | r = -0.085, p = 0.022 (11 models) |
| **Confidence ratio** | 1.688, CI [1.546, 1.844] contains phi = 1.618 |
| **Padilla prediction rate** | 0.418, CI [0.373, 0.463] contains 1/phi^2 |

## Reproduce

```bash
pip install -r requirements.txt

# Local models only (no API keys needed):
python run_experiment.py

# With all models (add API keys):
GOOGLE_API_KEY=... OPENAI_API_KEY=... python run_experiment.py

# Generate paper PDF:
python generate_paper.py
```

## Data

- **Behavioral data**: Stuyck et al. (2022), [OSF: osf.io/sc5n7](https://osf.io/sc5n7)
- **CRA triads**: Stuyck et al. (2021, Appendix B), [OSF: osf.io/snb3k](https://osf.io/snb3k)
- 106 participants, 4,482 trials, 69 items with complete triads (3 hint words + solution)

## Method

For each CRA item (e.g., *palm / slag / stam* -> *boom*):

1. Embed all 3 hint words and the solution across 11 models
2. Compute the L2-normalized centroid of the 3 hint-word embeddings
3. Measure cosine distance from centroid to solution embedding (d_centroid)
4. Correlate d_centroid with behavioral insight metrics from 106 human participants

## Files

```
run_experiment.py       # Complete experiment (single script)
generate_paper.py       # Paper PDF generator
requirements.txt        # Dependencies
data/
  stuyck_data.csv       # Behavioral data (Stuyck et al. 2022)
  crat_triads.csv       # 76 CRA triads (Stuyck et al. 2021)
figures/
  fig1_triple_convergence.png
  fig2_cascade.png
  fig3_meta_analysis.png
  fig4_band_replication.png
  fig5_cognitive_ratios.png
  fig6_theoretical_derivation.png
  fig7_scatter.png
  fig_main_dashboard.png
results/
  results.json           # All numerical results
  correlations.csv       # Full correlation table
```

## Models Tested

| Band | Models |
|---|---|
| 1/phi^4 (0.146) | E5-small, E5-base, GTE-small, GTE-base, Gemini-001 |
| 1/phi^2 (0.382) | BGE-base (0.384, **delta = 0.002**), BGE-small |
| 1/phi (0.618) | MiniLM-L6, MPNet, OpenAI-small, OpenAI-large |

## References

- Azpiroz, B. (2026). The phi constant of knowledge: Cross-domain geometric invariance in semantic embedding spaces.
- Stuyck, H., et al. (2021). The Aha! moment: Is insight a different form of problem solving? *Consciousness and Cognition*, 90, 103106.
- Stuyck, H., et al. (2022). Aha! under pressure. *Consciousness and Cognition*, 98, 103265.
- Grigoryan, A., & Grigoryan, M. (2025). Golden ratio in multidimensional spaces. *Mathematics*, 13(5), 699.
- Padilla, L. M., et al. (2026). The golden partition. arXiv:2602.15266.
- He, B., Xu, J., & Cheng, G. (2025). Golden ratio weighting prevents model collapse. arXiv:2502.18049.
- Jaeger, H. (2022). The golden ratio in machine learning. IEEE AIPR Workshop. arXiv:2006.04751.

## License

MIT
