# High-Frequency Detail Sampling based on Res Sampling

This is a ComfyUI custom node that enhances fine detail preservation in diffusion model outputs using spectral high-frequency emphasis (HFE).

## Installation

Clone or copy this folder into your ComfyUI `custom_nodes` directory:

```
ComfyUI/
  custom_nodes/
    RES4SHO/
      __init__.py
      sampling.py
```

Restart ComfyUI. The new samplers and schedulers will appear in the dropdown menus of any **KSampler** node.

## Samplers

All samplers are exponential integrators with phi-function coefficients. The HFE enhancement extracts high-frequency detail from inter-stage correction deltas via a 3x3 spatial high-pass filter and re-injects it with configurable strength.

### Fixed-Strength Presets

Each stage count offers 8 strength levels (`s1` = no emphasis, `s8` = maximum potential sharpness):

| Sampler | Stages | Model Evals/Step |
|---------|--------|-----------------|
| `hfe_s1` .. `hfe_s8` | 2 | 2 |
| `hfe3_s1` .. `hfe3_s8` | 3 | 3 |
| `hfe4_s1` .. `hfe4_s8` | 4 | 4 |
| `hfe5_s1` .. `hfe5_s8` | 5 | 5 |

Higher stage counts provide better ODE integration accuracy at the cost of more model evaluations per step.

### Adaptive (Auto) Samplers

Per-step adaptive `eta` based on sigma envelope and content gating:

| Sampler | Stages | Description |
|---------|--------|-------------|
| `hfe_auto` | 2 | Variable c2, eta, and kernel per step |
| `hfe3_auto` | 3 | Per-step eta with 3-stage integrator |
| `hfe4_auto` | 4 | Per-step eta with 4-stage integrator |
| `hfe5_auto` | 5 | Per-step eta with 5-stage integrator |

**How auto adapts:**
- **Sigma envelope** (smoothstep): suppresses emphasis at high noise (early steps), full strength in the detail-forming range
- **Content gate**: reduces emphasis when the model correction is already HF-rich; increases it when the correction is smooth and needs boosting

### Experimental Samplers (hfx_*)

Alternative HF extraction methods, all using a 2-stage base:

| Sampler | Method |
|---------|--------|
| `hfx_lap` | Laplacian pyramid multi-scale (3 bands) |
| `hfx_mom` | Correction momentum (EMA across steps) |
| `hfx_fft` | FFT spectral high-pass with smooth cutoff |
| `hfx_sde` | Stochastic HF noise injection |
| `hfx_spatial` | Spatially-adaptive per-pixel gating |

**Hybrids** (combine two techniques):
- `hfx_lap_mom` -- Laplacian pyramid + momentum
- `hfx_lap_spatial` -- Laplacian pyramid + spatial gating
- `hfx_fft_spatial` -- FFT spectral + spatial gating

**Band profile variants:**
- `hfx_lap_fine` -- fine-detail emphasis (edges, texture)
- `hfx_lap_broad` -- even emphasis across frequency bands

Each experimental mode also has 4 graduated strength presets (`_s1` .. `_s4`), e.g. `hfx_lap_s1`, `hfx_mom_s3`, etc.

## Schedulers

Arctangent S-curve schedulers that concentrate step density in the detail-forming sigma range:

| Scheduler | Description |
|-----------|-------------|
| `atan_gentle` | Mild mid-sigma concentration |
| `atan_focused` | Moderate detail-range concentration |
| `atan_steep` | Aggressive detail-range concentration |
| `karras_tan` | Karras-Tangent hybrid (experimental) |
| `logistic` | Logistic sigmoid S-curve (experimental) |

An ASCII sigma chart is printed to the console when a scheduler is used.

## Recommended Combinations

### Getting Started

| Goal | Sampler | Scheduler | Notes |
|------|---------|-----------|-------|
| General use | `hfe_auto` | `atan_focused` | Best all-rounder -- adaptive emphasis handles most content |
| Subtle enhancement | `hfe_s3` | `atan_gentle` | Light touch, minimal risk of artifacts |
| Strong detail | `hfe_s6` | `atan_steep` | Noticeably sharper textures and edges |
| Maximum sharpness | `hfe_s7` or `hfe_s8` | `atan_steep` | Aggressive -- inspect for over-sharpening |

### By Content Type

| Content | Sampler | Scheduler | Why |
|---------|---------|-----------|-----|
| Portraits / faces | `hfe_auto` | `atan_focused` | Auto gate protects smooth skin while sharpening eyes, hair, pores |
| Landscapes / nature | `hfe_s5` | `atan_gentle` | Fixed mid-strength avoids over-enhancing skies and gradients |
| Architecture / hard surfaces | `hfe_s7` | `atan_steep` | Strong emphasis on edges and geometric detail |
| Text / UI renders | `hfx_lap_fine` | `atan_steep` | Fine-band Laplacian targets glyph edges specifically |
| Fabric / organic texture | `hfx_lap_broad` | `atan_focused` | Even multi-scale emphasis across weave and folds |
| Illustrations / anime | `hfe_s4` | `atan_gentle` | Light emphasis preserves flat shading without adding unwanted texture |

### High-Accuracy Integrators

More model evaluations per step for better ODE integration -- useful at low step counts or with difficult models:

| Sampler | Scheduler | Use Case |
|---------|-----------|----------|
| `hfe3_auto` | `atan_focused` | Good balance of accuracy and speed (3 evals/step) |
| `hfe4_auto` | `atan_focused` | High accuracy for complex prompts (4 evals/step) |
| `hfe5_auto` | `atan_gentle` | Maximum integration accuracy (5 evals/step) |
| `hfe4_s5` | `atan_steep` | Fixed-strength detail + 4-stage accuracy |
| `hfe5_s6` | `karras_tan` | High emphasis + high accuracy + Karras hybrid spacing |

### Experimental Combinations

| Sampler | Scheduler | Character |
|---------|-----------|-----------|
| `hfx_lap` | `atan_focused` | Multi-scale detail -- good default experimental choice |
| `hfx_fft` | `atan_steep` | Frequency-domain sharpening -- clean spectral separation |
| `hfx_spatial` | `atan_focused` | Sharpens high-variance regions, leaves smooth areas alone |
| `hfx_mom` | `atan_gentle` | Accumulates detail across steps -- builds up gradually |
| `hfx_sde` | `atan_gentle` | Stochastic texture injection -- adds micro-variation |
| `hfx_lap_mom` | `atan_focused` | Multi-scale + momentum -- rich progressive detail |
| `hfx_lap_spatial` | `atan_steep` | Multi-scale + spatial gating -- targeted sharpening |
| `hfx_fft_spatial` | `atan_focused` | Spectral + spatial -- precise frequency-aware gating |

### Scheduler Pairings

| Scheduler | Best With | Character |
|-----------|-----------|-----------|
| `atan_gentle` | Low-strength samplers (`s1`-`s4`), stochastic modes | Mild concentration, safe for all content |
| `atan_focused` | Auto samplers, mid-strength presets (`s4`-`s6`) | Balanced step density in detail range |
| `atan_steep` | High-strength samplers (`s6`-`s8`), architectural content | Aggressive detail-range concentration |
| `karras_tan` | High-stage integrators (`hfe4_*`, `hfe5_*`) | Karras optimal spacing + tangent warp |
| `logistic` | Any -- alternative S-curve shape | Sharper transition through detail range, flatter extremes |

## How It Works

**Base integrator:** Multi-stage singlestep exponential integrator (res_Ns) with phi-function coefficients, giving exact treatment of exponential decay and higher-order corrections from intermediate evaluations.

**HFE enhancement:** The inter-stage correction delta captures what the model reveals at lower noise -- texture, edges, micro-structure. A spatial high-pass (residual after box blur in latent space) extracts the fine detail component, which is re-injected with extra weight `eta`. This compounds across every step.

**Cost:** One 3x3 `avg_pool` per step for all variants (negligible vs. model evaluation). Auto samplers add a few scalar ops on top.

## License

MIT
