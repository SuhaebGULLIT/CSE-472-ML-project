# Prosody-Guided Cross-Attention and Bottleneck Adapters for Cross-Corpus Bangla SER

> The first systematic cross-corpus speech emotion recognition (CCSER) study for Bangla. Training on **SUBESCO** and adapting to **BANGLASER** with frozen-backbone bottleneck adapters and prosody-guided cross-attention over Wav2Vec2.

📄 **[Read the paper](./2005075_2005086_ml_project.pdf)**

---

## TL;DR

Cross-corpus SER systems usually fail when moving between datasets, even within the same language. Prior work (AFTL) handled this by **unfreezing top transformer layers** during adaptation — which damages the very pretrained representations needed for cross-corpus generalization. We replace this with **frozen-backbone bottleneck adapters** and get a **+16.2 UA** improvement under identical data conditions.

| Protocol | V1 (AFTL-style) | V4 (Ours, single) | V5 Ensemble |
|---|---|---|---|
| Train-from-scratch (Main) | 72.79% UA | **80.52% UA** | **84.51% UA** |
| Zero-shot transfer (NA) | 47.92% UA | **63.59% UA** | — |
| Domain-adapted (DA) | 61.23% UA | **78.66% UA** | — |

All results: 5-fold speaker-independent cross-validation on BANGLASER.

---

## Why this matters

Bangla has **230M+ native speakers** and is severely underrepresented in speech emotion research. Despite SUBESCO and BANGLASER both being Bangla corpora recorded at 16 kHz in controlled settings, naïve transfer between them collapses to **47.9% UA** (barely above 25% chance for 4 classes). This work establishes the first benchmark and shows what actually closes the gap.

---

## Key Contributions

1. **First Bangla CCSER benchmark** across three protocols: zero-shot, train-from-scratch, and adapter-based domain adaptation.
2. **Frozen-backbone bottleneck adapters** (768→64→768) inserted after selected Wav2Vec2 blocks — preserves pretrained acoustic representations, adds only ~0.5% trainable parameters, and outperforms layer-unfreezing by **+16.2 UA**.
3. **Prosody-guided cross-attention**: projects the 103-dim prosody vector into **4 parallel query tokens** that attend over 115 acoustic frames, enabling multi-view prosody-acoustic alignment instead of single-query gating.
4. **Multi-layer learnable Wav2Vec2 aggregation** across blocks {1, 4, 7, 9, 12}, spanning phonetic → lexical → prosodic → emotion-discriminative → semantic levels.
5. **Structured 5-version ablation** with paired t-tests isolating each design decision; adapter-based DA is statistically significant at p<0.01.

---

## Architecture Overview

```
                  ┌──────────────────┐         ┌─────────────────────┐
Raw waveform ───▶ │ Wav2Vec2 (frozen)│         │   Prosody (103-d)   │
   (7s @ 16kHz)   │  + adapters (DA) │         │     Parselmouth     │
                  └────────┬─────────┘         └──────────┬──────────┘
                           │                              │
                  Blocks {1,4,7,9,12}              MLP 103→256→64
                           │                              │
                  Per-block Conv1d (768→64)        Multi-query proj
                           │                       (4 tokens × 64)
                  Learnable block aggregation             │
                           │                              │
                           ▼                              ▼
                       Acoustic (B×115×64) ◀─── Cross-attn (4 heads)
                                                  │
                                          Concat + Self-attn (8 heads)
                                                  │
                                          Attentive stats pooling
                                                  │
                                          AM-Softmax (4 classes)
```

See Figure 1 in the paper for the full diagram.

---

## Headline Result: Adapter vs. Layer-Unfreezing

The single most important finding of this work:

| Adaptation strategy | DA UA on BANGLASER |
|---|---|
| Layer-unfreeze (AFTL, V1) | 61.23% |
| **Frozen-backbone adapters (V3)** | **78.29%** |

**Same data, same epochs, same source checkpoint — only the adaptation strategy changes.** Adapters preserve what Wav2Vec2 learned during pretraining; unfreezing erases it.

---

## Datasets

| Corpus | Utterances | Speakers | Emotions used |
|---|---|---|---|
| [SUBESCO](https://doi.org/10.5281/zenodo.4526477) (source) | 4,000 | 20 (10F, 10M) | happy, sad, angry, neutral |
| [BANGLASER](https://data.mendeley.com/datasets/t9h6p943xy/) (target) | 864 | 30 (15F, 15M) | happy, sad, angry, neutral |

---



> **Note:** Code release is in progress. Open an issue if you'd like to reproduce specific results.

---

## Setup

```bash
# Python 3.10+, CUDA 12.x recommended
pip install torch transformers parselmouth-praat numpy scikit-learn pyyaml
```

Backbone: `facebook/wav2vec2-base` (95M params, 12 transformer blocks).

Hardware used for training: NVIDIA H100 80GB. Wav2Vec2-base also runs on a single 16GB GPU with batch size 4.

---

## Reproducing Key Numbers

```bash
# Train V4 source model on SUBESCO (5-fold)
python src/train_source.py --config configs/v4.yaml

# Adapter-based domain adaptation to BANGLASER
python src/adapt.py --config configs/v4.yaml --source-ckpt runs/v4/source.pt

# Evaluate
python src/evaluate.py --config configs/v4.yaml --protocol DA
```

---

## Limitations

- Target corpus (BANGLASER) is small (864 utterances), which inflates per-fold variance.
- Only 4 emotions evaluated (happy, sad, angry, neutral) due to corpus overlap.
- V3 introduces cross-attention and adapters together; we attribute contributions theoretically (NA gain ⇒ cross-attn, DA gain ⇒ adapters) but a clean factorial ablation is future work.

---

## Citation

If this work is useful to you:

```bibtex
@misc{matin-uddin-2026-bangla-ccser,
  title  = {Prosody-Guided Cross-Attention and Bottleneck Adapters for
            Cross-Corpus Speech Emotion Recognition in Low-Resource Bangla},
  author = {Matin, Suhaeb Bin and Uddin, Moyen},
  year   = {2026},
  note   = {Bangladesh University of Engineering and Technology}
}
```

---

## Acknowledgments

Bangladesh University of Engineering and Technology. Builds on Wav2Vec2 (Baevski et al., 2020), AFTL (Naderi & Nasersharif, 2023), and bottleneck adapters (Houlsby et al., 2019).
