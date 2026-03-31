#!/usr/bin/env python3
"""
The Phi Signature in Human Cognition
=====================================
Complete experiment: CRA triads embedded across multiple models,
correlated with human insight behavior.

Usage:
    python run_experiment.py                    # local models only
    GOOGLE_API_KEY=... OPENAI_API_KEY=... python run_experiment.py  # all models

Inputs (in data/):
    stuyck_data.csv   — Behavioral data from Stuyck et al. (2022), OSF: osf.io/sc5n7
    crat_triads.csv   — Complete CRA triads from Stuyck et al. (2021, Appendix B)

Outputs (in results/):
    results.json      — All numerical results
    correlations.csv  — Full correlation table

Figures (in figures/):
    fig1_triple_convergence.png
    fig2_cascade.png
    fig3_meta_analysis.png
    fig4_band_replication.png
    fig5_cognitive_ratios.png
    fig6_theoretical_derivation.png
    fig7_robustness.png
    fig_main_dashboard.png

Author: Borja Azpiroz, 2026
Companion paper: "The phi constant of knowledge" (Azpiroz, 2026)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2        # 1.6180...
PHI_INV = 1 / PHI                  # 0.6180...
PHI_INV2 = 1 / PHI**2              # 0.3820...
PHI_INV3 = 1 / PHI**3              # 0.2361...
PHI_INV4 = 1 / PHI**4              # 0.1459...
YOUDEN_V9 = 0.386                   # From companion paper v9

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGURES_DIR = os.path.join(ROOT, 'figures')
for d in [RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

# Plot style
COLORS = {
    'phi': '#C9A227', 'insight': '#2196F3', 'analytic': '#FF5722',
    'bg': '#1a1a2e', 'text': '#e0e0e0', 'grid': '#333355',
    'green': '#4CAF50', 'purple': '#9C27B0', 'orange': '#FF9800',
    'red': '#F44336', 'T1': '#4CAF50', 'T2': '#2196F3',
    'T3': '#FF9800', 'T4': '#F44336',
}

def style_ax(ax):
    ax.set_facecolor(COLORS['bg'])
    ax.tick_params(colors=COLORS['text'])
    for s in ax.spines.values():
        s.set_color(COLORS['grid'])


# ═════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═════════════════════════════════════════════════════════════
def load_data():
    """Load behavioral data and CRA triads, merge them."""
    print("=" * 70)
    print("  SECTION 1: Loading data")
    print("=" * 70)

    df = pd.read_csv(os.path.join(DATA_DIR, 'stuyck_data.csv'),
                      sep=';', encoding='utf-8-sig')
    df = df.dropna(subset=['RT', 'confidence', 'ACC_CRAT', 'solution_type'])
    df = df[(df['RT'] > 0) & (df['RT'] < df['RT'].quantile(0.99))]

    triads = pd.read_csv(os.path.join(DATA_DIR, 'crat_triads.csv'),
                          encoding='utf-8')

    # Per-item behavioral statistics
    items = df.groupby('CRAT_correct').agg(
        n_trials=('RT', 'count'),
        insight_rate=('solution_type', lambda x: (x == 1).mean()),
        mean_rt=('RT', 'mean'),
        mean_conf=('confidence', 'mean'),
        accuracy=('ACC_CRAT', 'mean'),
    ).reset_index()
    items = items[items['n_trials'] >= 10]

    # Merge triads with behavioral data
    merged = triads.merge(items, left_on='solution', right_on='CRAT_correct',
                           how='inner')

    print(f"  Trials: {len(df)}")
    print(f"  Subjects: {df['subject'].nunique()}")
    print(f"  CRA triads: {len(triads)}")
    print(f"  Matched items (triads with >= 10 trials): {len(merged)}")
    print(f"  Insight trials: {(df['solution_type']==1).sum()} "
          f"({(df['solution_type']==1).mean():.1%})")

    return df, merged


# ═════════════════════════════════════════════════════════════
# SECTION 2: EMBEDDING
# ═════════════════════════════════════════════════════════════
def embed_all_models(merged):
    """Embed CRA triads across all available models.

    For each item: embed 3 hint words + solution.
    Compute centroid of hints, then cosine distance centroid -> solution.
    """
    print("\n" + "=" * 70)
    print("  SECTION 2: Embedding CRA triads")
    print("=" * 70)

    w1 = merged['word1'].tolist()
    w2 = merged['word2'].tolist()
    w3 = merged['word3'].tolist()
    sol = merged['solution'].tolist()
    n = len(merged)

    all_models = {}

    # --- Local models (sentence-transformers) ---
    from sentence_transformers import SentenceTransformer

    local_specs = [
        ('BAAI/bge-base-en-v1.5', 'BGE-base'),
        ('BAAI/bge-small-en-v1.5', 'BGE-small'),
        ('sentence-transformers/all-MiniLM-L6-v2', 'MiniLM-L6'),
        ('sentence-transformers/all-mpnet-base-v2', 'MPNet'),
        ('intfloat/e5-small-v2', 'E5-small'),
        ('intfloat/e5-base-v2', 'E5-base'),
        ('thenlper/gte-small', 'GTE-small'),
        ('thenlper/gte-base', 'GTE-base'),
    ]

    for model_id, name in local_specs:
        print(f"  {name}...", end=' ', flush=True)
        try:
            model = SentenceTransformer(model_id)
            e1 = model.encode(w1, normalize_embeddings=True, show_progress_bar=False)
            e2 = model.encode(w2, normalize_embeddings=True, show_progress_bar=False)
            e3 = model.encode(w3, normalize_embeddings=True, show_progress_bar=False)
            es = model.encode(sol, normalize_embeddings=True, show_progress_bar=False)

            # Centroid of 3 hint words
            centroid = (e1 + e2 + e3) / 3
            centroid /= np.linalg.norm(centroid, axis=1, keepdims=True)

            # Distances
            d_centroid = np.array([cosine(centroid[i], es[i]) for i in range(n)])
            d_mean = np.mean([
                np.array([cosine(e1[i], es[i]) for i in range(n)]),
                np.array([cosine(e2[i], es[i]) for i in range(n)]),
                np.array([cosine(e3[i], es[i]) for i in range(n)]),
            ], axis=0)

            all_models[name] = {
                'd_centroid': d_centroid,
                'd_mean': d_mean,
                'dims': e1.shape[1],
                'provider': 'sentence-transformers',
            }
            print(f"OK ({e1.shape[1]}d) d_centroid={d_centroid.mean():.4f}")
            del model
        except Exception as e:
            print(f"FAILED: {e}")

    # --- Gemini API ---
    gemini_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if gemini_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=gemini_key)
            cfg = types.EmbedContentConfig(
                task_type='SEMANTIC_SIMILARITY', output_dimensionality=3072)

            def gemini_embed(texts, batch_size=20):
                embs = []
                for i in range(0, len(texts), batch_size):
                    r = client.models.embed_content(
                        model='gemini-embedding-001',
                        contents=texts[i:i+batch_size], config=cfg)
                    for e in r.embeddings:
                        embs.append(e.values)
                    time.sleep(0.3)
                arr = np.array(embs)
                arr /= np.linalg.norm(arr, axis=1, keepdims=True)
                return arr

            print(f"  Gemini-001...", end=' ', flush=True)
            e1 = gemini_embed(w1)
            e2 = gemini_embed(w2)
            e3 = gemini_embed(w3)
            es = gemini_embed(sol)
            centroid = (e1 + e2 + e3) / 3
            centroid /= np.linalg.norm(centroid, axis=1, keepdims=True)
            dc = np.array([cosine(centroid[i], es[i]) for i in range(n)])
            dm = np.mean([
                np.array([cosine(e1[i], es[i]) for i in range(n)]),
                np.array([cosine(e2[i], es[i]) for i in range(n)]),
                np.array([cosine(e3[i], es[i]) for i in range(n)]),
            ], axis=0)
            all_models['Gemini-001'] = {
                'd_centroid': dc, 'd_mean': dm,
                'dims': 3072, 'provider': 'google',
            }
            print(f"OK (3072d) d_centroid={dc.mean():.4f}")
        except Exception as e:
            print(f"  Gemini: {e}")
    else:
        print("  Gemini: no API key (set GOOGLE_API_KEY)")

    # --- OpenAI API ---
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            for oai_model in ['text-embedding-3-small', 'text-embedding-3-large']:
                short = oai_model.split('-')[-1]
                mname = f'OpenAI-{short}'
                print(f"  {mname}...", end=' ', flush=True)

                def oai_embed(texts):
                    r = client.embeddings.create(input=texts, model=oai_model)
                    arr = np.array([e.embedding for e in r.data])
                    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
                    return arr

                e1 = oai_embed(w1)
                e2 = oai_embed(w2)
                e3 = oai_embed(w3)
                es = oai_embed(sol)
                centroid = (e1 + e2 + e3) / 3
                centroid /= np.linalg.norm(centroid, axis=1, keepdims=True)
                dc = np.array([cosine(centroid[i], es[i]) for i in range(n)])
                dm = np.mean([
                    np.array([cosine(e1[i], es[i]) for i in range(n)]),
                    np.array([cosine(e2[i], es[i]) for i in range(n)]),
                    np.array([cosine(e3[i], es[i]) for i in range(n)]),
                ], axis=0)
                all_models[mname] = {
                    'd_centroid': dc, 'd_mean': dm,
                    'dims': e1.shape[1], 'provider': 'openai',
                }
                print(f"OK ({e1.shape[1]}d) d_centroid={dc.mean():.4f}")
        except Exception as e:
            print(f"  OpenAI: {e}")
    else:
        print("  OpenAI: no API key (set OPENAI_API_KEY)")

    print(f"\n  Total models: {len(all_models)}")
    return all_models


# ═════════════════════════════════════════════════════════════
# SECTION 3: ANALYSIS
# ═════════════════════════════════════════════════════════════

def analyze_bands(all_models):
    """Assign each model to a phi-power band based on d_centroid."""
    print("\n" + "=" * 70)
    print("  SECTION 3A: Band analysis")
    print("=" * 70)

    phi_bands = [
        ('1/phi^4', PHI_INV4),
        ('1/phi^3', PHI_INV3),
        ('1/phi^2', PHI_INV2),
        ('1/phi', PHI_INV),
    ]
    band_results = {}

    for name in sorted(all_models.keys()):
        d = all_models[name]['d_centroid'].mean()
        closest = min(phi_bands, key=lambda x: abs(x[1] - d))
        delta = abs(closest[1] - d)
        band_results[name] = {
            'mean_d': d, 'band': closest[0],
            'band_value': closest[1], 'delta': delta,
        }
        print(f"  {name:20s}: {d:.4f} -> {closest[0]} (delta={delta:.4f})")

    return band_results


def analyze_bootstrap(all_models):
    """Bootstrap CI for BGE-base d_centroid and triple convergence test."""
    print("\n" + "=" * 70)
    print("  SECTION 3B: Bootstrap CI and triple convergence")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)
    results = {}

    for name in ['BGE-base', 'BGE-small', 'GTE-base', 'GTE-small']:
        if name not in all_models:
            continue
        dc = all_models[name]['d_centroid']
        boots = [np.mean(dc[np.random.choice(len(dc), len(dc), replace=True)])
                 for _ in range(N_BOOTSTRAP)]
        ci_l, ci_h = np.percentile(boots, [2.5, 97.5])
        contains_phi2 = ci_l <= PHI_INV2 <= ci_h
        contains_youden = ci_l <= YOUDEN_V9 <= ci_h

        results[name] = {
            'mean': dc.mean(), 'ci_low': ci_l, 'ci_high': ci_h,
            'contains_phi2': contains_phi2,
            'contains_youden': contains_youden,
            'delta_phi2': abs(dc.mean() - PHI_INV2),
            'boots': boots,
        }
        print(f"  {name}: {dc.mean():.6f}, CI [{ci_l:.6f}, {ci_h:.6f}]")
        print(f"    1/phi^2={PHI_INV2:.6f} in CI: {contains_phi2}")
        print(f"    Youden={YOUDEN_V9} in CI: {contains_youden}")

    # Triple convergence
    if 'BGE-base' in results:
        bge = results['BGE-base']['mean']
        max_delta = max(abs(PHI_INV2 - bge), abs(YOUDEN_V9 - bge),
                        abs(PHI_INV2 - YOUDEN_V9))
        print(f"\n  TRIPLE CONVERGENCE:")
        print(f"    1/phi^2:      {PHI_INV2:.6f}")
        print(f"    BGE centroid:  {bge:.6f} (delta={abs(bge-PHI_INV2):.6f})")
        print(f"    Youden v9:     {YOUDEN_V9:.6f} (delta={abs(YOUDEN_V9-PHI_INV2):.6f})")
        print(f"    All within:    {max_delta:.4f}")

    return results


def analyze_cascade(df, merged, all_models):
    """Classify trials and test the four-group cascade."""
    print("\n" + "=" * 70)
    print("  SECTION 3C: Cascade replication")
    print("=" * 70)

    # Classify trials
    subj_med_rt = df.groupby('subject')['RT'].median().to_dict()
    subj_med_conf = df.groupby('subject')['confidence'].median().to_dict()

    types = []
    for _, row in df.iterrows():
        ins = row['solution_type'] == 1
        cor = row['ACC_CRAT'] == 1
        fast = row['RT'] < subj_med_rt[row['subject']]
        conf = row['confidence'] > subj_med_conf[row['subject']]
        if not cor:
            types.append('T4_noise')
        elif ins and fast and conf:
            types.append('T1_internal')
        elif ins and cor:
            types.append('T2_integrator')
        else:
            types.append('T3_pseudo')
    df['cascade_type'] = types

    # Trial counts
    counts = df['cascade_type'].value_counts()
    for t in ['T1_internal', 'T2_integrator', 'T3_pseudo', 'T4_noise']:
        n = counts.get(t, 0)
        print(f"  {t}: {n} ({n/len(df):.1%})")

    # Per-item fractions
    fracs = df.groupby('CRAT_correct').agg(
        f_internal=('cascade_type', lambda x: (x == 'T1_internal').mean()),
        f_integrator=('cascade_type', lambda x: (x == 'T2_integrator').mean()),
        f_pseudo=('cascade_type', lambda x: (x == 'T3_pseudo').mean()),
        f_noise=('cascade_type', lambda x: (x == 'T4_noise').mean()),
    ).reset_index()
    mdata = merged.merge(fracs, left_on='solution', right_on='CRAT_correct',
                          how='inner')

    # Test cascade per model
    all_d_int = []
    all_d_noise = []
    cascade_per_model = {}

    for name in sorted(all_models.keys()):
        dc = all_models[name]['d_centroid']
        if len(dc) != len(mdata):
            continue
        vals = {}
        for ftype in ['f_internal', 'f_integrator', 'f_pseudo', 'f_noise']:
            w = mdata[ftype].values
            if w.sum() > 0:
                vals[ftype] = np.average(dc, weights=w)

        if len(vals) == 4:
            full = (vals['f_internal'] < vals['f_integrator'] <
                    vals['f_pseudo'] < vals['f_noise'])
            int_lt_noise = vals['f_internal'] < vals['f_noise']
            all_d_int.append(vals['f_internal'])
            all_d_noise.append(vals['f_noise'])
            cascade_per_model[name] = {
                'd_internal': vals['f_internal'],
                'd_integrator': vals['f_integrator'],
                'd_pseudo': vals['f_pseudo'],
                'd_noise': vals['f_noise'],
                'full_cascade': full,
                'int_lt_noise': int_lt_noise,
            }
            label = 'FULL' if full else 'PARTIAL' if int_lt_noise else 'NO'
            print(f"  {name:20s}: {vals['f_internal']:.4f} < "
                  f"{vals['f_integrator']:.4f} < {vals['f_pseudo']:.4f} < "
                  f"{vals['f_noise']:.4f} [{label}]")

    n_models = len(cascade_per_model)
    n_full = sum(1 for v in cascade_per_model.values() if v['full_cascade'])
    n_lt = sum(1 for v in cascade_per_model.values() if v['int_lt_noise'])
    t_stat, p_val = stats.ttest_rel(all_d_int, all_d_noise) if len(all_d_int) > 2 else (0, 1)

    print(f"\n  Full cascade: {n_full}/{n_models}")
    print(f"  d(Internal) < d(Noise): {n_lt}/{n_models}")
    print(f"  Paired t = {t_stat:.3f}, p = {p_val:.6f}")

    return {
        'cascade_per_model': cascade_per_model,
        'n_full': n_full, 'n_models': n_models,
        'n_int_lt_noise': n_lt,
        't_stat': t_stat, 'p_val': p_val,
        'mdata': mdata,
    }


def analyze_correlations(merged, all_models):
    """Correlate d_centroid with behavioral variables + meta-analysis."""
    print("\n" + "=" * 70)
    print("  SECTION 3D: Correlations and meta-analysis")
    print("=" * 70)

    behavioral = {
        'insight_rate': merged['insight_rate'].values,
        'mean_rt': merged['mean_rt'].values,
        'mean_conf': merged['mean_conf'].values,
        'accuracy': merged['accuracy'].values,
    }

    all_corrs = []
    meta_rs = []
    meta_ns = []
    meta_names = []

    for name in sorted(all_models.keys()):
        dc = all_models[name]['d_centroid']
        if len(dc) != len(merged):
            continue

        for bname, bvals in behavioral.items():
            mask = np.isfinite(dc) & np.isfinite(bvals)
            if mask.sum() < 10:
                continue
            r, p = stats.pearsonr(dc[mask], bvals[mask])
            rho, p_rho = stats.spearmanr(dc[mask], bvals[mask])
            all_corrs.append({
                'model': name, 'distance': 'd_centroid', 'variable': bname,
                'pearson_r': r, 'pearson_p': p,
                'spearman_rho': rho, 'spearman_p': p_rho, 'n': int(mask.sum()),
            })
            if bname == 'insight_rate':
                meta_rs.append(r)
                meta_ns.append(mask.sum())
                meta_names.append(name)
                sig = '*' if p < 0.05 else ''
                print(f"  {name:20s} x insight: r={r:+.4f} (p={p:.4f}){sig}")

    corr_df = pd.DataFrame(all_corrs)

    # FDR correction
    if len(corr_df) > 0:
        ps = corr_df['pearson_p'].values
        n_tests = len(ps)
        sorted_idx = np.argsort(ps)
        p_adj = np.ones(n_tests)
        for rank, idx in enumerate(sorted_idx, 1):
            p_adj[idx] = ps[idx] * n_tests / rank
        for i in range(len(sorted_idx) - 2, -1, -1):
            p_adj[sorted_idx[i]] = min(p_adj[sorted_idx[i]],
                                        p_adj[sorted_idx[i + 1]])
        corr_df['p_adj_fdr'] = np.minimum(p_adj, 1.0)

    # Meta-analysis
    meta_rs = np.array(meta_rs)
    meta_ns = np.array(meta_ns)
    zs = np.arctanh(meta_rs)
    weights = meta_ns - 3
    z_w = np.sum(zs * weights) / np.sum(weights)
    se = 1 / np.sqrt(np.sum(weights))
    z_stat = z_w / se
    p_meta = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    r_meta = np.tanh(z_w)
    ci_meta_l = np.tanh(z_w - 1.96 * se)
    ci_meta_h = np.tanh(z_w + 1.96 * se)
    n_neg = int((meta_rs < 0).sum())

    print(f"\n  META-ANALYSIS (d_centroid x insight_rate):")
    print(f"    Combined r = {r_meta:.4f}, CI [{ci_meta_l:.4f}, {ci_meta_h:.4f}]")
    print(f"    z = {z_stat:.3f}, p = {p_meta:.6f}")
    print(f"    Direction (r<0): {n_neg}/{len(meta_rs)}")

    # Permutation test for BGE-base
    perm_result = {}
    if 'BGE-base' in all_models:
        np.random.seed(RANDOM_SEED)
        dc_bge = all_models['BGE-base']['d_centroid']
        insight = merged['insight_rate'].values
        r_obs, _ = stats.pearsonr(dc_bge, insight)
        perm_rs = [stats.pearsonr(dc_bge, np.random.permutation(insight))[0]
                   for _ in range(N_PERMUTATIONS)]
        p_perm = np.mean(np.abs(perm_rs) >= np.abs(r_obs))
        perm_result = {'r': r_obs, 'p_perm': p_perm, 'perm_rs': perm_rs}
        print(f"    BGE-base permutation: r={r_obs:.4f}, p={p_perm:.4f}")

    n_fdr = int((corr_df['p_adj_fdr'] < 0.05).sum()) if 'p_adj_fdr' in corr_df else 0
    print(f"    FDR survivors (alpha=0.05): {n_fdr}/{len(corr_df)}")

    meta_result = {
        'r': r_meta, 'ci_low': ci_meta_l, 'ci_high': ci_meta_h,
        'z': z_stat, 'p': p_meta,
        'n_negative': n_neg, 'n_models': len(meta_rs),
        'individual_rs': meta_rs.tolist(),
        'model_names': meta_names,
    }

    return corr_df, meta_result, perm_result


def analyze_cognitive_ratios(df):
    """Test phi in behavioral ratios (confidence, RT, Padilla prediction rate)."""
    print("\n" + "=" * 70)
    print("  SECTION 3E: Cognitive ratios")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    # Per-subject ratios
    subj = df.groupby('subject').apply(lambda g: pd.Series({
        'conf_insight': g[g['solution_type'] == 1]['confidence'].mean(),
        'conf_analytic': g[g['solution_type'] == 2]['confidence'].mean(),
        'rt_insight': g[g['solution_type'] == 1]['RT'].mean(),
        'rt_analytic': g[g['solution_type'] == 2]['RT'].mean(),
        'n_insight': (g['solution_type'] == 1).sum(),
        'n_analytic': (g['solution_type'] == 2).sum(),
    })).dropna()
    subj = subj[(subj['n_insight'] >= 5) & (subj['n_analytic'] >= 5)]

    conf_ratio = subj['conf_insight'] / subj['conf_analytic']
    rt_ratio = subj['rt_insight'] / subj['rt_analytic']

    # Bootstrap CIs
    boot_conf = [np.random.choice(conf_ratio.values, len(conf_ratio),
                                   replace=True).mean() for _ in range(N_BOOTSTRAP)]
    ci_conf = np.percentile(boot_conf, [2.5, 97.5])

    boot_rt = [np.random.choice(rt_ratio.values, len(rt_ratio),
                                 replace=True).mean() for _ in range(N_BOOTSTRAP)]
    ci_rt = np.percentile(boot_rt, [2.5, 97.5])

    # Padilla prediction rate
    subj_med_conf = df.groupby('subject')['confidence'].median().to_dict()
    df['is_prediction'] = (df['ACC_CRAT'] == 1) & df.apply(
        lambda r: r['confidence'] > subj_med_conf.get(r['subject'], 50), axis=1)
    pred_rate_items = df.groupby('CRAT_correct')['is_prediction'].mean()
    pred_rate_items = pred_rate_items[pred_rate_items.index.isin(
        df.groupby('CRAT_correct').size()[lambda x: x >= 10].index)]

    boot_pred = [np.random.choice(pred_rate_items.values, len(pred_rate_items),
                                   replace=True).mean() for _ in range(N_BOOTSTRAP)]
    ci_pred = np.percentile(boot_pred, [2.5, 97.5])

    results = {
        'conf_ratio': {'mean': conf_ratio.mean(), 'ci': ci_conf.tolist(),
                        'contains_phi': ci_conf[0] <= PHI <= ci_conf[1],
                        'boots': boot_conf},
        'rt_ratio': {'mean': rt_ratio.mean(), 'ci': ci_rt.tolist()},
        'padilla_prediction': {'mean': pred_rate_items.mean(), 'ci': ci_pred.tolist(),
                                'contains_phi2': ci_pred[0] <= PHI_INV2 <= ci_pred[1],
                                'boots': boot_pred},
    }

    print(f"  Conf insight/analytic: {conf_ratio.mean():.4f}, "
          f"CI [{ci_conf[0]:.4f}, {ci_conf[1]:.4f}], "
          f"phi={PHI:.4f} in CI: {results['conf_ratio']['contains_phi']}")
    print(f"  RT insight/analytic: {rt_ratio.mean():.4f}, "
          f"CI [{ci_rt[0]:.4f}, {ci_rt[1]:.4f}]")
    print(f"  Padilla prediction rate: {pred_rate_items.mean():.4f}, "
          f"CI [{ci_pred[0]:.4f}, {ci_pred[1]:.4f}], "
          f"1/phi^2={PHI_INV2:.4f} in CI: {results['padilla_prediction']['contains_phi2']}")

    return results


def analyze_robustness(merged, all_models):
    """LOO stability and Cook's distance for key models."""
    print("\n" + "=" * 70)
    print("  SECTION 3F: Robustness (LOO, Cook's D)")
    print("=" * 70)

    insight = merged['insight_rate'].values
    rob = {}

    for name in ['BGE-base', 'GTE-base']:
        if name not in all_models:
            continue
        dc = all_models[name]['d_centroid']
        if len(dc) != len(merged):
            continue

        r_full, p_full = stats.pearsonr(dc, insight)

        # LOO
        loo_rs = []
        for i in range(len(dc)):
            d_loo = np.delete(dc, i)
            i_loo = np.delete(insight, i)
            r_loo, _ = stats.pearsonr(d_loo, i_loo)
            loo_rs.append(r_loo)

        loo_rs = np.array(loo_rs)
        sign_changes = int(np.sum(np.sign(loo_rs) != np.sign(r_full)))
        max_influence = float(np.max(np.abs(loo_rs - r_full)))

        # Cook's distance
        from numpy.linalg import lstsq
        X = np.column_stack([np.ones(len(dc)), dc])
        beta, _, _, _ = lstsq(X, insight, rcond=None)
        y_hat = X @ beta
        resid = insight - y_hat
        mse = np.sum(resid**2) / (len(dc) - 2)
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        h = np.diag(H)
        cooks_d = (resid**2 / (2 * mse)) * (h / (1 - h)**2)
        n_outliers = int(np.sum(cooks_d > 4 / len(dc)))

        rob[name] = {
            'r_full': r_full, 'loo_min': loo_rs.min(), 'loo_max': loo_rs.max(),
            'sign_changes': sign_changes, 'max_influence': max_influence,
            'n_outliers': n_outliers,
        }
        print(f"  {name}: LOO [{loo_rs.min():.4f}, {loo_rs.max():.4f}], "
              f"sign changes: {sign_changes}, outliers: {n_outliers}")

    return rob


# ═════════════════════════════════════════════════════════════
# SECTION 4: FIGURES
# ═════════════════════════════════════════════════════════════

def make_fig1_triple_convergence(bootstrap_res):
    """Triple convergence: 1/phi^2, BGE centroid, Youden."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['bg'])

    if 'BGE-base' not in bootstrap_res:
        plt.close()
        return

    bge = bootstrap_res['BGE-base']

    # Left: Bootstrap distribution
    ax = axes[0]; style_ax(ax)
    ax.hist(bge['boots'], bins=80, color=COLORS['insight'], alpha=0.7, density=True)
    ax.axvline(bge['ci_low'], color='white', linewidth=1.5, linestyle='--')
    ax.axvline(bge['ci_high'], color='white', linewidth=1.5, linestyle='--')
    ax.axvline(PHI_INV2, color=COLORS['phi'], linewidth=3,
               label=f'1/phi^2 = {PHI_INV2:.4f}')
    ax.axvline(YOUDEN_V9, color=COLORS['green'], linewidth=2.5, linestyle='--',
               label=f'Youden = {YOUDEN_V9}')
    ax.set_title(f'Bootstrap CI: [{bge["ci_low"]:.4f}, {bge["ci_high"]:.4f}]\n'
                 f'Contains 1/phi^2: {bge["contains_phi2"]} | Youden: {bge["contains_youden"]}',
                 color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])
    ax.set_xlabel('BGE-base d_centroid', color=COLORS['text'])

    # Right: Bar chart
    ax = axes[1]; style_ax(ax)
    vals = [PHI_INV2, bge['mean'], YOUDEN_V9]
    names = ['1/phi^2\n(theory)', 'BGE centroid\n(CRA triads)', 'Youden\n(paper v9)']
    cols = [COLORS['phi'], COLORS['insight'], COLORS['green']]
    ax.barh(range(3), vals, color=cols, alpha=0.8, height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.0003, i, f'{v:.4f}', va='center',
                color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.set_yticks(range(3))
    ax.set_yticklabels(names, color=COLORS['text'], fontsize=10)
    ax.set_xlim(0.374, 0.396)
    ax.set_title('TRIPLE CONVERGENCE\ndelta < 0.004',
                 color=COLORS['phi'], fontsize=13, fontweight='bold')

    fig.suptitle('Figure 1: Triple Convergence at 1/phi^2 = 0.382',
                 color=COLORS['phi'], fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_triple_convergence.png'),
                dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()


def make_fig2_cascade(cascade_res, all_models):
    """Cascade replication figure."""
    cpm = cascade_res['cascade_per_model']
    if not cpm:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['bg'])

    # Left: BGE-base cascade bars
    ax = axes[0]; style_ax(ax)
    if 'BGE-base' in cpm:
        bge = cpm['BGE-base']
        vals = [bge['d_internal'], bge['d_integrator'],
                bge['d_pseudo'], bge['d_noise']]
        names = ['Internal', 'Integrator', 'Pseudo', 'Noise']
        cols = [COLORS['T1'], COLORS['T2'], COLORS['T3'], COLORS['T4']]
        ax.barh(range(4), vals, color=cols, alpha=0.8, height=0.6)
        for i, v in enumerate(vals):
            ax.text(v + 0.0003, i, f'{v:.4f}', va='center',
                    color=COLORS['text'], fontsize=10)
        ax.set_yticks(range(4))
        ax.set_yticklabels(names, color=COLORS['text'], fontsize=10)
        ax.axvline(PHI_INV2, color=COLORS['phi'], linewidth=1.5, linestyle='--',
                   label=f'1/phi^2')
        ax.legend(fontsize=9, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
                  labelcolor=COLORS['text'])
    ax.set_title(f'BGE-base Cascade (triads)\n'
                 f'd(Internal) < d(Noise): {cascade_res["n_int_lt_noise"]}/{cascade_res["n_models"]}, '
                 f'p = {cascade_res["p_val"]:.5f}',
                 color=COLORS['text'], fontsize=11, fontweight='bold')

    # Right: All models Internal vs Noise
    ax = axes[1]; style_ax(ax)
    models_sorted = sorted(cpm.keys(),
                            key=lambda m: cpm[m]['d_internal'])
    y_pos = range(len(models_sorted))
    d_ints = [cpm[m]['d_internal'] for m in models_sorted]
    d_nois = [cpm[m]['d_noise'] for m in models_sorted]
    ax.barh([y - 0.15 for y in y_pos], d_ints, height=0.3,
            color=COLORS['T1'], alpha=0.8, label='Internal')
    ax.barh([y + 0.15 for y in y_pos], d_nois, height=0.3,
            color=COLORS['T4'], alpha=0.8, label='Noise')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(models_sorted, fontsize=7, color=COLORS['text'])
    ax.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])
    ax.set_title('All models: Internal vs Noise',
                 color=COLORS['text'], fontsize=11, fontweight='bold')

    fig.suptitle('Figure 2: Cascade Replication',
                 color=COLORS['phi'], fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_cascade.png'),
                dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()


def make_fig3_meta_analysis(meta_result, perm_result):
    """Meta-analysis forest plot + permutation test."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['bg'])

    # Left: Forest plot
    ax = axes[0]; style_ax(ax)
    rs = meta_result['individual_rs']
    names = meta_result['model_names']
    sorted_idx = np.argsort(rs)
    rs_sorted = [rs[i] for i in sorted_idx]
    names_sorted = [names[i] for i in sorted_idx]
    cols = [COLORS['green'] if r < 0 else COLORS['analytic'] for r in rs_sorted]
    ax.barh(range(len(rs_sorted)), rs_sorted, color=cols, alpha=0.7, height=0.5)
    ax.axvline(0, color=COLORS['grid'], linewidth=1)
    ax.axvline(meta_result['r'], color=COLORS['phi'], linewidth=2.5,
               label=f'Meta r={meta_result["r"]:.3f} (p={meta_result["p"]:.4f})')
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=7, color=COLORS['text'])
    ax.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])
    ax.set_title('Meta-analysis: r(d_centroid, insight_rate)',
                 color=COLORS['text'], fontsize=11, fontweight='bold')

    # Right: Permutation distribution
    ax = axes[1]; style_ax(ax)
    if perm_result:
        ax.hist(perm_result['perm_rs'], bins=80, color=COLORS['grid'],
                alpha=0.7, density=True, label='Null distribution')
        ax.axvline(perm_result['r'], color=COLORS['phi'], linewidth=3,
                   label=f'Observed r={perm_result["r"]:.3f}')
        ax.set_title(f'BGE-base permutation test\np = {perm_result["p_perm"]:.4f}',
                     color=COLORS['text'], fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
                  labelcolor=COLORS['text'])

    fig.suptitle('Figure 3: Meta-Analysis and Permutation Test',
                 color=COLORS['phi'], fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_meta_analysis.png'),
                dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()


def make_fig4_bands(band_results, all_models):
    """Band replication across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    style_ax(ax)

    models_sorted = sorted(band_results.keys(),
                            key=lambda m: band_results[m]['mean_d'])
    y_pos = range(len(models_sorted))
    vals = [band_results[m]['mean_d'] for m in models_sorted]

    ax.barh(list(y_pos), vals, color=COLORS['insight'], alpha=0.7, height=0.6)
    for name, val in [('1/phi^4', PHI_INV4), ('1/phi^3', PHI_INV3),
                       ('1/phi^2', PHI_INV2), ('1/phi', PHI_INV)]:
        ax.axvline(val, color=COLORS['phi'], linewidth=2, linestyle='--')
        ax.text(val, len(models_sorted) - 0.3, name, color=COLORS['phi'],
                fontsize=9, ha='center', fontweight='bold')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(models_sorted, fontsize=8, color=COLORS['text'])
    ax.set_title('Figure 4: Phi-Power Band Replication (d_centroid, triads)',
                 color=COLORS['phi'], fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean cosine distance (centroid -> solution)', color=COLORS['text'])

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_band_replication.png'),
                dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()


def make_fig5_cognitive_ratios(ratio_res):
    """Confidence ratio and Padilla prediction rate bootstrap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['bg'])

    # Confidence ratio
    ax = axes[0]; style_ax(ax)
    cr = ratio_res['conf_ratio']
    ax.hist(cr['boots'], bins=80, color=COLORS['purple'], alpha=0.7, density=True)
    ax.axvline(PHI, color=COLORS['phi'], linewidth=3, label=f'phi = {PHI:.3f}')
    ax.axvline(cr['ci'][0], color='white', linewidth=1.5, linestyle='--')
    ax.axvline(cr['ci'][1], color='white', linewidth=1.5, linestyle='--')
    ax.set_title(f'Confidence ratio insight/analytic\n'
                 f'CI [{cr["ci"][0]:.3f}, {cr["ci"][1]:.3f}], '
                 f'phi in CI: {cr["contains_phi"]}',
                 color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    # Padilla prediction rate
    ax = axes[1]; style_ax(ax)
    pr = ratio_res['padilla_prediction']
    ax.hist(pr['boots'], bins=80, color=COLORS['insight'], alpha=0.7, density=True)
    ax.axvline(PHI_INV2, color=COLORS['phi'], linewidth=3,
               label=f'1/phi^2 = {PHI_INV2:.3f}')
    ax.axvline(pr['ci'][0], color='white', linewidth=1.5, linestyle='--')
    ax.axvline(pr['ci'][1], color='white', linewidth=1.5, linestyle='--')
    ax.set_title(f'Padilla prediction rate\n'
                 f'CI [{pr["ci"][0]:.3f}, {pr["ci"][1]:.3f}], '
                 f'1/phi^2 in CI: {pr["contains_phi2"]}',
                 color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    fig.suptitle('Figure 5: Phi in Cognitive Ratios',
                 color=COLORS['phi'], fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_cognitive_ratios.png'),
                dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()


def make_fig6_theoretical():
    """Theoretical derivation: 4 convergent paths to p^2 + p - 1 = 0."""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.patch.set_facecolor(COLORS['bg'])

    # P1: Universal quadratic
    ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
    x = np.linspace(-2, 2, 500)
    ax.plot(x, x**2 + x - 1, color=COLORS['phi'], linewidth=2.5)
    ax.axhline(0, color=COLORS['grid'], linewidth=0.5)
    p_phi = (np.sqrt(5) - 1) / 2
    ax.axvline(p_phi, color=COLORS['insight'], linewidth=1.5, linestyle='--',
               label=f'p = 1/phi = {p_phi:.4f}')
    ax.scatter([p_phi], [0], color=COLORS['phi'], s=100, zorder=5)
    ax.set_title('p^2 + p - 1 = 0\nThe universal equation',
                 color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    # P2: Padilla balance function
    ax = fig.add_subplot(gs[0, 1]); style_ax(ax)
    p_range = np.linspace(0.01, 0.99, 500)
    f_vals = [-(1-p)*np.log(1-p) + np.log(p) for p in p_range]
    ax.plot(p_range, f_vals, color=COLORS['green'], linewidth=2.5)
    ax.axvline(p_phi, color=COLORS['phi'], linewidth=2, linestyle='--',
               label=f'Self-similarity: p=1/phi')
    ax.axhline(0, color=COLORS['grid'], linewidth=0.5)
    ax.set_title('Prediction/surprise balance\n(Padilla et al. 2026)',
                 color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.set_xlabel('p (prediction)', color=COLORS['text'])
    ax.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    # P3: He et al. error function
    ax = fig.add_subplot(gs[0, 2]); style_ax(ax)
    w_range = np.linspace(0.01, 0.99, 500)
    C_vals = [(w**2 + (1-w)**2) / (2*w - w**2) for w in w_range]
    ax.plot(w_range, C_vals, color=COLORS['analytic'], linewidth=2.5)
    w_opt = p_phi
    C_opt = (w_opt**2 + (1-w_opt)**2) / (2*w_opt - w_opt**2)
    ax.scatter([w_opt], [C_opt], color=COLORS['phi'], s=150, zorder=5,
               label=f'w* = 1/phi = {w_opt:.4f}')
    ax.set_title('Error C(w,1): signal/noise mixing\n(He et al. 2025)',
                 color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 3)
    ax.legend(fontsize=9, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    # P4: Grigoryan GGR
    ax = fig.add_subplot(gs[1, 0]); style_ax(ax)
    alpha = np.linspace(0, np.pi, 500)
    ggr = []
    for a in alpha:
        coeffs = [1, 0, -1, -2*np.cos(a), -1]
        roots = np.roots(coeffs)
        real_pos = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
        ggr.append(max(real_pos) if real_pos else np.nan)
    ax.plot(np.degrees(alpha), ggr, color=COLORS['phi'], linewidth=2.5)
    ax.axhline(PHI, color=COLORS['insight'], linewidth=1, linestyle=':',
               label=f'phi = {PHI:.4f}')
    ax.axhline(PHI_INV, color=COLORS['analytic'], linewidth=1, linestyle=':',
               label=f'1/phi = {PHI_INV:.4f}')
    ax.set_title('Generalized Golden Ratio\n(Grigoryan & Grigoryan 2025)',
                 color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Angle (degrees)', color=COLORS['text'])
    ax.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    # P5: Phi bands
    ax = fig.add_subplot(gs[1, 1]); style_ax(ax)
    band_colors = ['#E91E63', '#FF9800', '#FFEB3B', '#8BC34A']
    band_vals = [PHI_INV4, PHI_INV3, PHI_INV2, PHI_INV, 1.0]
    zone_names = ['Deep\nintegration', 'Genuine\nintegration',
                  'Pseudo-\nintegration', 'Structured\nnoise']
    for i in range(4):
        ax.axhspan(band_vals[i], band_vals[i+1], alpha=0.3, color=band_colors[i])
        mid = (band_vals[i] + band_vals[i+1]) / 2
        ax.text(0.5, mid, zone_names[i], ha='center', va='center',
                color=COLORS['text'], fontsize=9, fontweight='bold')
    for name, val in [('1/phi^4', PHI_INV4), ('1/phi^3', PHI_INV3),
                       ('1/phi^2', PHI_INV2), ('1/phi', PHI_INV)]:
        ax.axhline(val, color=COLORS['phi'], linewidth=1.5)
        ax.text(0.95, val + 0.01, f'{name}={val:.3f}', ha='right',
                color=COLORS['phi'], fontsize=8, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title('Phi bands in cosine distance',
                 color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.set_xticks([])

    # P6: Convergence diagram
    ax = fig.add_subplot(gs[1, 2]); style_ax(ax)
    sources = ['Grigoryan\n(Vector\ngeometry)', 'Padilla\n(Prediction/\nSurprise)',
               'Jaeger\n(KL/Shannon\nBalance)', 'He et al.\n(Signal/Noise\nMixing)']
    angles_pos = [np.pi/2 + i * np.pi/2 for i in range(4)]
    for i, (src, ang) in enumerate(zip(sources, angles_pos)):
        x_pos = 0.6 * np.cos(ang)
        y_pos = 0.6 * np.sin(ang)
        ax.annotate(src, (x_pos, y_pos), ha='center', va='center',
                    fontsize=7, color=COLORS['text'], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'],
                             edgecolor=band_colors[i], linewidth=2))
        ax.annotate('', xy=(0, 0), xytext=(x_pos * 0.5, y_pos * 0.5),
                    arrowprops=dict(arrowstyle='->', color=band_colors[i], linewidth=2))
    circle = plt.Circle((0, 0), 0.15, color=COLORS['phi'], alpha=0.8)
    ax.add_patch(circle)
    ax.text(0, 0, 'phi', ha='center', va='center', fontsize=20,
            color=COLORS['bg'], fontweight='bold')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
    ax.set_title('Four independent convergences',
                 color=COLORS['text'], fontsize=11, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Figure 6: Theoretical Derivation - Why Phi Emerges',
                 color=COLORS['phi'], fontsize=14, fontweight='bold', y=1.01)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig6_theoretical_derivation.png'),
                dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()


def make_fig7_scatter(merged, all_models):
    """Scatter plot: BGE-base d_centroid vs insight_rate."""
    if 'BGE-base' not in all_models:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    style_ax(ax)

    dc = all_models['BGE-base']['d_centroid']
    insight = merged['insight_rate'].values

    ax.scatter(dc, insight, c=insight, cmap='RdYlGn_r', s=60, alpha=0.8,
               edgecolors='white', linewidth=0.5)

    z = np.polyfit(dc, insight, 1)
    xl = np.linspace(dc.min(), dc.max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), color=COLORS['phi'], linewidth=2.5, linestyle='--')

    ax.axvline(PHI_INV2, color=COLORS['phi'], linewidth=1.5, linestyle=':',
               alpha=0.7, label=f'1/phi^2 = {PHI_INV2:.3f}')

    r, p = stats.pearsonr(dc, insight)
    ax.set_title(f'BGE-base d_centroid vs insight rate\n'
                 f'r = {r:.3f}, p = {p:.4f}',
                 color=COLORS['text'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Cosine distance (centroid of 3 hints -> solution)',
                  color=COLORS['text'])
    ax.set_ylabel('Insight rate', color=COLORS['text'])
    ax.legend(fontsize=9, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    fig.suptitle('Figure 7: Embedding Distance Predicts Insight',
                 color=COLORS['phi'], fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig7_scatter.png'),
                dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()


def make_dashboard(merged, all_models, bootstrap_res, cascade_res,
                    meta_result, perm_result, ratio_res, band_results):
    """Main dashboard figure combining key results."""
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.patch.set_facecolor(COLORS['bg'])

    dc_bge = all_models.get('BGE-base', {}).get('d_centroid')
    insight = merged['insight_rate'].values
    bge_boot = bootstrap_res.get('BGE-base', {})

    # P1: Scatter
    ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
    if dc_bge is not None:
        ax.scatter(dc_bge, insight, c=insight, cmap='RdYlGn_r', s=50, alpha=0.8,
                   edgecolors='white', linewidth=0.3)
        z = np.polyfit(dc_bge, insight, 1)
        xl = np.linspace(dc_bge.min(), dc_bge.max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color=COLORS['phi'], linewidth=2, linestyle='--')
        r, p = stats.pearsonr(dc_bge, insight)
        perm_p = perm_result.get('p_perm', 'N/A')
        ax.set_title(f'BGE centroid vs insight\nr={r:.3f}, perm p={perm_p}',
                     color=COLORS['text'], fontsize=10, fontweight='bold')
    ax.set_xlabel('d_centroid', color=COLORS['text'])
    ax.set_ylabel('Insight rate', color=COLORS['text'])

    # P2: Bootstrap CI
    ax = fig.add_subplot(gs[0, 1]); style_ax(ax)
    if bge_boot:
        ax.hist(bge_boot['boots'], bins=80, color=COLORS['insight'], alpha=0.7, density=True)
        ax.axvline(bge_boot['ci_low'], color='white', linewidth=1.5, linestyle='--')
        ax.axvline(bge_boot['ci_high'], color='white', linewidth=1.5, linestyle='--')
        ax.axvline(PHI_INV2, color=COLORS['phi'], linewidth=3, label=f'1/phi^2')
        ax.axvline(YOUDEN_V9, color=COLORS['green'], linewidth=2.5, linestyle='--', label='Youden')
        ax.set_title(f'Bootstrap CI [{bge_boot["ci_low"]:.4f}, {bge_boot["ci_high"]:.4f}]',
                     color=COLORS['text'], fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, facecolor=COLORS['bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])

    # P3: Triple convergence
    ax = fig.add_subplot(gs[0, 2]); style_ax(ax)
    if bge_boot:
        vals = [PHI_INV2, bge_boot['mean'], YOUDEN_V9]
        names = ['1/phi^2', 'BGE centroid', 'Youden v9']
        cols = [COLORS['phi'], COLORS['insight'], COLORS['green']]
        ax.barh(range(3), vals, color=cols, alpha=0.8, height=0.5)
        for i, v in enumerate(vals):
            ax.text(v + 0.0003, i, f'{v:.4f}', va='center', color=COLORS['text'], fontsize=10, fontweight='bold')
        ax.set_yticks(range(3)); ax.set_yticklabels(names, color=COLORS['text'], fontsize=9)
        ax.set_xlim(0.374, 0.396)
    ax.set_title('TRIPLE CONVERGENCE', color=COLORS['phi'], fontsize=12, fontweight='bold')

    # P4: Cascade
    ax = fig.add_subplot(gs[1, 0]); style_ax(ax)
    cpm = cascade_res['cascade_per_model']
    if 'BGE-base' in cpm:
        bge_c = cpm['BGE-base']
        vals = [bge_c['d_internal'], bge_c['d_integrator'], bge_c['d_pseudo'], bge_c['d_noise']]
        cols = [COLORS['T1'], COLORS['T2'], COLORS['T3'], COLORS['T4']]
        ax.barh(range(4), vals, color=cols, alpha=0.8, height=0.6)
        for i, v in enumerate(vals): ax.text(v+0.0003, i, f'{v:.4f}', va='center', color=COLORS['text'], fontsize=9)
        ax.set_yticks(range(4)); ax.set_yticklabels(['Internal','Integrator','Pseudo','Noise'], color=COLORS['text'])
        ax.axvline(PHI_INV2, color=COLORS['phi'], linewidth=1.5, linestyle='--')
    ax.set_title(f'Cascade p={cascade_res["p_val"]:.5f}', color=COLORS['text'], fontsize=10, fontweight='bold')

    # P5: Band replication
    ax = fig.add_subplot(gs[1, 1]); style_ax(ax)
    ms = sorted(band_results.keys(), key=lambda m: band_results[m]['mean_d'])
    ax.barh(range(len(ms)), [band_results[m]['mean_d'] for m in ms],
            color=COLORS['insight'], alpha=0.7, height=0.6)
    for n, v in [('1/phi^4',PHI_INV4),('1/phi^2',PHI_INV2),('1/phi',PHI_INV)]:
        ax.axvline(v, color=COLORS['phi'], linewidth=2, linestyle='--')
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=6, color=COLORS['text'])
    ax.set_title('Band replication (d_centroid)', color=COLORS['text'], fontsize=10, fontweight='bold')

    # P6: Confidence ratio
    ax = fig.add_subplot(gs[1, 2]); style_ax(ax)
    cr = ratio_res['conf_ratio']
    ax.hist(cr['boots'], bins=80, color=COLORS['purple'], alpha=0.7, density=True)
    ax.axvline(PHI, color=COLORS['phi'], linewidth=3, label=f'phi={PHI:.3f}')
    ax.axvline(cr['ci'][0], color='white', linewidth=1, linestyle='--')
    ax.axvline(cr['ci'][1], color='white', linewidth=1, linestyle='--')
    ax.set_title(f'Conf ratio CI [{cr["ci"][0]:.3f},{cr["ci"][1]:.3f}]',
                 color=COLORS['text'], fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])

    # P7: Meta forest
    ax = fig.add_subplot(gs[2, 0]); style_ax(ax)
    rs = meta_result['individual_rs']; mnames = meta_result['model_names']
    si = np.argsort(rs)
    cols = [COLORS['green'] if rs[i]<0 else COLORS['analytic'] for i in si]
    ax.barh(range(len(si)), [rs[i] for i in si], color=cols, alpha=0.7, height=0.5)
    ax.axvline(meta_result['r'], color=COLORS['phi'], linewidth=2.5)
    ax.set_yticks(range(len(si))); ax.set_yticklabels([mnames[i] for i in si], fontsize=6, color=COLORS['text'])
    ax.set_title(f'Meta r={meta_result["r"]:.3f}, p={meta_result["p"]:.4f}',
                 color=COLORS['text'], fontsize=10, fontweight='bold')

    # P8: d_centroid distribution
    ax = fig.add_subplot(gs[2, 1]); style_ax(ax)
    if dc_bge is not None:
        ax.hist(dc_bge, bins=20, color=COLORS['insight'], alpha=0.7, density=True)
        for n, v in [('1/phi^2',PHI_INV2)]:
            ax.axvline(v, color=COLORS['phi'], linewidth=2, linestyle='--')
    ax.set_title('BGE-base d_centroid distribution', color=COLORS['text'], fontsize=10, fontweight='bold')

    # P9: Summary
    ax = fig.add_subplot(gs[2, 2]); style_ax(ax); ax.axis('off')
    bge_mean = bge_boot.get('mean', 0) if bge_boot else 0
    summary = (
        f"FINAL RESULTS - TRIADS ONLY\n"
        f"{'='*35}\n\n"
        f"Items: {len(merged)} CRA triads\n"
        f"Models: {len(all_models)}\n\n"
        f"TRIPLE CONVERGENCE:\n"
        f"  1/phi^2 = {PHI_INV2:.4f}\n"
        f"  BGE     = {bge_mean:.4f} (d={abs(bge_mean-PHI_INV2):.4f})\n"
        f"  Youden  = {YOUDEN_V9}\n\n"
        f"CASCADE: {cascade_res['n_int_lt_noise']}/{cascade_res['n_models']}\n"
        f"  p = {cascade_res['p_val']:.6f}\n\n"
        f"META: r={meta_result['r']:.3f}, p={meta_result['p']:.4f}\n"
        f"BGE perm: p={perm_result.get('p_perm','N/A')}\n\n"
        f"CONF RATIO: {cr['mean']:.3f}\n"
        f"  phi in CI: {cr['contains_phi']}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace', color=COLORS['text'],
            bbox=dict(boxstyle='round', facecolor=COLORS['bg'],
                      edgecolor=COLORS['phi'], linewidth=2))

    title = (f'THE PHI SIGNATURE IN HUMAN COGNITION - COMPLETE CRA TRIADS\n'
             f'BGE centroid = {bge_mean:.4f} | delta = {abs(bge_mean-PHI_INV2):.4f} from 1/phi^2 | '
             f'Cascade {cascade_res["n_int_lt_noise"]}/{cascade_res["n_models"]} '
             f'p={cascade_res["p_val"]:.5f} | Meta p={meta_result["p"]:.4f}')
    fig.suptitle(title, color=COLORS['phi'], fontsize=12, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_main_dashboard.png'),
                dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()


# ═════════════════════════════════════════════════════════════
# SECTION 5: SAVE RESULTS
# ═════════════════════════════════════════════════════════════

def save_results(band_results, bootstrap_res, cascade_res, corr_df,
                  meta_result, perm_result, ratio_res, robustness):
    """Save all numerical results to JSON and CSV."""

    # Clean bootstrap boots for JSON serialization
    boot_clean = {}
    for name, res in bootstrap_res.items():
        boot_clean[name] = {k: v for k, v in res.items() if k != 'boots'}

    ratio_clean = {}
    for k, v in ratio_res.items():
        ratio_clean[k] = {kk: vv for kk, vv in v.items() if kk != 'boots'}

    perm_clean = {k: v for k, v in perm_result.items() if k != 'perm_rs'}

    cascade_clean = {k: v for k, v in cascade_res.items() if k != 'mdata'}

    results = {
        'experiment': 'The Phi Signature in Human Cognition',
        'author': 'Borja Azpiroz, 2026',
        'phi': PHI,
        'phi_inv2': PHI_INV2,
        'youden_v9': YOUDEN_V9,
        'band_analysis': band_results,
        'bootstrap': boot_clean,
        'cascade': cascade_clean,
        'meta_analysis': meta_result,
        'permutation_bge': perm_clean,
        'cognitive_ratios': ratio_clean,
        'robustness': robustness,
    }

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    corr_df.to_csv(os.path.join(RESULTS_DIR, 'correlations.csv'), index=False)
    print(f"\n  Saved results.json and correlations.csv to {RESULTS_DIR}")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "#" * 70)
    print("#  THE PHI SIGNATURE IN HUMAN COGNITION")
    print("#  Complete experiment with CRA triads")
    print("#" * 70 + "\n")

    # 1. Load
    df, merged = load_data()

    # 2. Embed
    all_models = embed_all_models(merged)

    if len(all_models) == 0:
        print("\nERROR: No models loaded. Check your environment.")
        sys.exit(1)

    # 3. Analyze
    band_results = analyze_bands(all_models)
    bootstrap_res = analyze_bootstrap(all_models)
    cascade_res = analyze_cascade(df, merged, all_models)
    corr_df, meta_result, perm_result = analyze_correlations(merged, all_models)
    ratio_res = analyze_cognitive_ratios(df)
    robustness = analyze_robustness(merged, all_models)

    # 4. Figures
    print("\n" + "=" * 70)
    print("  SECTION 4: Generating figures")
    print("=" * 70)

    make_fig1_triple_convergence(bootstrap_res)
    print("  fig1_triple_convergence.png")

    make_fig2_cascade(cascade_res, all_models)
    print("  fig2_cascade.png")

    make_fig3_meta_analysis(meta_result, perm_result)
    print("  fig3_meta_analysis.png")

    make_fig4_bands(band_results, all_models)
    print("  fig4_band_replication.png")

    make_fig5_cognitive_ratios(ratio_res)
    print("  fig5_cognitive_ratios.png")

    make_fig6_theoretical()
    print("  fig6_theoretical_derivation.png")

    make_fig7_scatter(merged, all_models)
    print("  fig7_scatter.png")

    make_dashboard(merged, all_models, bootstrap_res, cascade_res,
                    meta_result, perm_result, ratio_res, band_results)
    print("  fig_main_dashboard.png")

    # 5. Save
    save_results(band_results, bootstrap_res, cascade_res, corr_df,
                  meta_result, perm_result, ratio_res, robustness)

    # 6. Summary
    print("\n" + "#" * 70)
    print("#  EXPERIMENT COMPLETE")
    print("#" * 70)
    print(f"\n  Figures: {FIGURES_DIR}/")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Models tested: {len(all_models)}")
    if 'BGE-base' in bootstrap_res:
        bge = bootstrap_res['BGE-base']
        print(f"\n  KEY RESULT: BGE-base d_centroid = {bge['mean']:.6f}")
        print(f"  Bootstrap CI: [{bge['ci_low']:.6f}, {bge['ci_high']:.6f}]")
        print(f"  1/phi^2 = {PHI_INV2:.6f} in CI: {bge['contains_phi2']}")
        print(f"  Youden = {YOUDEN_V9} in CI: {bge['contains_youden']}")
    print(f"  Cascade: {cascade_res['n_int_lt_noise']}/{cascade_res['n_models']} "
          f"(p = {cascade_res['p_val']:.6f})")
    print(f"  Meta r = {meta_result['r']:.4f} (p = {meta_result['p']:.6f})")
    print()
