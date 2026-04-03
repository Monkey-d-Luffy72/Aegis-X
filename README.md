<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-RTX_3050+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Phi3:Mini-FF6F00?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Version-3.0_(Dual--Pipeline)-8B5CF6?style=for-the-badge" />
</p>

<h1 align="center">🛡️ Aegis-X</h1>
<h3 align="center">Advanced Deepfake Forensic Detection — Dual-Pipeline Architecture</h3>

<p align="center">
  <em>A multi-modal, agentic forensic system that orchestrates 10 specialized detection tools — spanning classical physics, frequency analysis, physiological signals, and deep neural networks — through an intelligent CPU→GPU dual-pipeline to deliver explainable, confidence-weighted verdicts on media authenticity.</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-dual-pipeline-flow">Pipeline</a> •
  <a href="#-tool-manifest">Tools</a> •
  <a href="#-key-changes-v30">What's New</a> •
  <a href="#-web-interface">Web UI</a> •
  <a href="#%EF%B8%8F-configuration">Config</a> •
  <a href="#-benchmarking">Benchmarks</a>
</p>

---

## ✨ Key Features

- **10-Tool Forensic Arsenal** — 6 CPU-bound classical/physics/signal tools + 4 GPU-accelerated neural networks, each targeting an independent manipulation vector
- **Dual-Pipeline with Face Gate** — Media is intelligently routed: a 4-dimension Face Gate decides whether the full bio-signal CPU path runs before GPU inference
- **CPU→GPU Early Stopping Gate** — After the CPU phase, a directional confidence gate (`HALT` / `MINIMAL_GPU` / `FULL_GPU`) prevents unnecessary GPU execution when the CPU already has consensus
- **30s CPU / 60s GPU Per-Tool Timeouts** — `ThreadPoolExecutor`-enforced timeouts on every tool; hung tools never block the pipeline
- **Centralized Error Factory** — `_make_error_result()` unifies all tool error `ToolResult` construction across CPU and GPU phases, ensuring structural parity
- **DRY VRAM Execution** — GPU tools use closure-safe `make_loader(t)` / `make_inference(data)` factory functions; no late-binding capture bugs
- **DEGRADED Mode Propagation** — If >50% of tools error out, the verdict dict and SSE events include `"degraded": true` for consumers
- **MediaPipe + CPU-SORT Tracking** — Bidirectional face tracking with Kalman filters; `max_confidence` derived from tracking coverage ratio (not hardcoded 1.0)
- **rPPG Temporal Window Safety** — `face_window = (0, 0)` now returns ABSTAIN instead of silently processing all frames as noise
- **Explainable AI Verdicts** — Every verdict is grounded in tool-level evidence summaries, passed to Ollama's Phi-3 Mini for natural-language forensic explanations
- **Real-Time Streaming UI** — Glassmorphic web interface with Server-Sent Events (SSE) for live tool-by-tool progress updates

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.10 |
| GPU VRAM | 3 GB | 6 GB+ |
| CUDA | 11.8+ | 12.1+ |
| RAM | 8 GB | 16 GB |
| Ollama | Installed with `phi3:mini` pulled | — |

### Automated Setup

```bash
# Clone the repository
git clone https://github.com/gaurav337/aegis-test.git
cd aegis-test

# Run the one-click installer
# (Creates venvs, installs dependencies, downloads weights from Kaggle)
python setup.py
```

> **Note:** You need a [Kaggle API token](https://www.kaggle.com/settings) at `~/.kaggle/kaggle.json` for automatic weight downloads.

### Manual Setup

```bash
# 1. Create virtual environments
python3.10 -m venv .venv_main
python3.10 -m venv .venv_gpu

# 2. Install dependencies
.venv_main/bin/pip install -r requirements-main.txt
.venv_gpu/bin/pip install -r requirements-gpu.txt

# 3. Download model weights from Kaggle
# https://www.kaggle.com/datasets/gauravkumarjangid/aegis-pth
# Place them as:
#   models/univfd/probe.pth
#   models/xception/xception_deepfake.pth
#   models/sbi/efficientnet_b4.pth
#   models/freqnet/cnndetect_resnet50.pth

# 4. Download CLIP backbone (auto-downloads on first run, or manually):
#   models/clip-vit-large-patch14/  (from HuggingFace: openai/clip-vit-large-patch14)

# 5. Ensure Ollama is running with Phi-3
ollama pull phi3:mini
ollama serve
```

### Launch

> **⚠️ Important:** Always use the explicit binary path to avoid environment cross-contamination:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./.venv_main/bin/python run_web.py
# Open http://localhost:8000
```

---

## 🏗 Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         WEB INTERFACE                           │
│                   (FastAPI + SSE Streaming)                     │
├─────────────────────────────────────────────────────────────────┤
│                       ForensicAgent                             │
│                                                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │   .venv_main            │  │   .venv_gpu (via proxy)      │  │
│  │   CPU PHASE             │  │   GPU PHASE                  │  │
│  │                         │  │                              │  │
│  │  [C2PA] → [DCT]        │  │  [FreqNet] → [UnivFD]       │  │
│  │  [rPPG] → [Geometry]   │  │  [Xception] → [SBI*]        │  │
│  │  [Illumin.] → [Corneal]│  │   *face-pipeline only        │  │
│  └──────────┬──────────────┘  └─────────────┬────────────────┘  │
│             │                               │                   │
│          ┌──▼──────────────────────────────▼──┐                │
│          │       CPU→GPU GATE                  │                │
│          │  HALT / MINIMAL_GPU / FULL_GPU       │                │
│          └──────────────────┬──────────────────┘                │
│                             │                                   │
│         ┌───────────────────▼───────────────────┐               │
│         │         EnsembleAggregator             │               │
│         │  (Directional confidence • Suspicion   │               │
│         │   Overdrive • Conflict detection)      │               │
│         └───────────────────┬───────────────────┘               │
│                             │                                   │
│         ┌───────────────────▼───────────────────┐               │
│         │          LLM Synthesis                 │               │
│         │       (Ollama / Phi-3 Mini)            │               │
│         └───────────────────┬───────────────────┘               │
│                             ▼                                   │
│                  { verdict, score, explanation,                 │
│                    degraded } → SSE → Browser                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Dual-Pipeline Flow

### Stage 0 — Preprocessing

```
Media Input (Image / Video)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│                  PREPROCESSOR                        │
│  (utils/preprocessing.py)                           │
│                                                     │
│  Video path:                                        │
│    extract_frames() → CPU-SORT Kalman tracking      │
│    Build trajectory_bboxes per tracked face         │
│    face_window = best contiguous run of frames      │
│    max_confidence = tracked_frames / total_frames   │ ← v3.0: was always 1.0
│    224×224 crop + 380×380 crop per identity         │
│    heuristic_flags: MOTION_BLUR, OCCLUSION, etc.   │
│                                                     │
│  Image path:                                        │
│    MediaPipe FaceMesh (static mode)                 │
│    Single face → TrackedFace dataclass              │
│    max_confidence = 1.0 (deterministic)             │
└──────────────────────┬──────────────────────────────┘
                       │ PreprocessResult
                       ▼
```

### Stage 1 — Face Gate (4 Dimensions)

```
                    PreprocessResult
                          │
        ┌─────────────────▼────────────────────┐
        │           FACE GATE                   │
        │  core/agent.py :: analyze()           │
        │                                       │
        │  ① has_face = True?                  │
        │  ② max_confidence ≥ 0.60?            │ ← tracking coverage ratio
        │  ③ max_face_area_ratio ≥ 0.01?       │
        │  ④ frames_with_faces_pct ≥ 0.30?     │
        │                                       │
        │  ALL 4 ✓ → pass_face_gate = True     │
        │  Any ✗  → pass_face_gate = False     │
        └───────┬───────────────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
  Face Pipeline      No-Face Pipeline
  (bio-signal tools  (frequency/spectral
   + SBI added)       tools only)
```

### Stage 2 — CPU Phase (Segment A)

```
pass_face_gate?
       │
  YES──┤  cpu_tools = [check_c2pa, run_dct, run_rppg,
       │               run_geometry, run_illumination,
       │               run_corneal]
       │
  NO───┤  cpu_tools = [check_c2pa, run_dct]
       │
       │  heuristic_flags may disable some tools:
       │    MOTION_BLUR/OCCLUSION → skip geo/illum/corneal
       │    FACE_TOO_SMALL        → skip corneal
       │    LOW_LIGHT             → skip illum/corneal
       │
       ▼
  For each tool:
    result = _safe_execute_tool(tool_name, input_data, timeout=30)
    ────────────────────────────────────────────────────
    ThreadPoolExecutor → future.result(timeout=30)
    On TimeoutError → _make_error_result("Timeout after 30s")
    On Exception    → _make_error_result(str(e))
    ────────────────────────────────────────────────────
    ensemble.add_result(result)
    yield AgentEvent("TOOL_COMPLETED")

  Special case: C2PA verified?
    → yield AgentEvent("EARLY_STOP")
    → return {"verdict": "REAL", "score": 1.0, "explanation": ...}
```

### Stage 3 — CPU→GPU Gate (Segment B)

```
After CPU phase completes...

    cpu_results = [successful, non-error, non-ABSTAIN results]
    decisive_results = [r for r if |r.score - 0.5| > 0.05]
                                   ↑
                         Filters out neutral-score (0.5)
                         tools that don't know direction

    if len(decisive_results) < 2:
        gate_decision = "FULL_GPU"   ← not enough signal
    else:
        Directional Confidence Math:
        ──────────────────────────────
        baseline_weights = {
            run_rppg: 0.35, run_geometry: 0.25,
            run_dct: 0.15, run_illumination: 0.10,
            run_corneal: 0.10, check_c2pa: 0.05
        }
        direction_i   = (score_i - 0.5) × 2  ∈ [-1.0, +1.0]
        agg_direction = Σ (direction_i × confidence_i × normalized_weight_i)
        agg_conf      = |agg_direction|

        Unison = all decisive_results agree on direction (REAL or FAKE)
        Domains = {bio, phys, freq, auth} from contributing tools

        if agg_conf > 0.93 AND unison AND |domains| ≥ 2:
            gate_decision = "HALT"         ← skip GPU entirely
        elif agg_conf ≥ 0.80:
            gate_decision = "MINIMAL_GPU"  ← one GPU tool only (UnivFD)
        else:
            gate_decision = "FULL_GPU"     ← full GPU sequence

    yield AgentEvent("GATE_DECISION", {decision, confidence, unison})
```

### Stage 4 — GPU Phase (Segment C)

```
gate_decision ≠ HALT?
       │
       ▼
  Build gpu_sequence:
    pass_face_gate:  [freqnet, univfd, xception, sbi]
    no face:         [freqnet, univfd, xception]
    MINIMAL_GPU:     [univfd]           ← override

  For each tool_name in gpu_sequence:
    tool = registry.get_tool(tool_name)
    req_vram = GPU_VRAM_REQUIREMENTS[tool_name]
    ──────────────────────────────────────────────────
    GPU_VRAM_REQUIREMENTS = {
        "run_freqnet": 0.4 GB
        "run_univfd":  0.6 GB
        "run_xception":0.5 GB
        "run_sbi":     0.8 GB
    }
    ──────────────────────────────────────────────────
    # Closure-safe factory (v3.0 fix)
    make_loader(tool)      → lambda: tool (bound by value)
    make_inference(data)   → lambda t: t.execute(data)

    ThreadPoolExecutor → run_with_vram_cleanup(
        make_loader(tool),
        make_inference(input_data),
        model_name=tool_name,
        required_vram_gb=req_vram,
        timeout=60                ← GPU timeout
    )

    On FuturesTimeoutError → _make_error_result("Timeout after 60s")
    On Exception           → _make_error_result(str(e))
    torch.cuda.empty_cache() in both error paths
```

### Stage 5 — DEGRADED Check + Ensemble + LLM

```
    is_degraded = total_errors / total_results > 0.50
    if is_degraded:
        logger.warning("DEGRADED: >50% tool failures")

    final_score  = ensemble.get_final_score()      ← 1.0 = REAL
    verdict_str  = ensemble.get_verdict()           ← "REAL" / "FAKE"
    explanation  = generate_verdict(...)            ← Ollama LLM

    yield AgentEvent("VERDICT", {
        verdict, score, explanation, degraded       ← v3.0: degraded propagated
    })

    return {
        "verdict": verdict_str,
        "score": final_score,
        "explanation": explanation,
        "degraded": is_degraded                     ← v3.0: consumers can detect
    }
```

---

## 🔬 Tool Manifest

### CPU Tools (`.venv_main` — Zero VRAM)

| Tool | File | Weight | Trust | Target Threat | Method |
|---|---|---|---|---|---|
| `check_c2pa` | `c2pa_tool.py` | 0.05 | Tier 1 | Provenance forgery | Cryptographic C2PA metadata chain |
| `run_dct` | `dct_tool.py` | 0.15 | Tier 1 | JPEG re-encoding | Double-quantization frequency peaks |
| `run_geometry` | `geometry_tool.py` | 0.18 | Tier 3 | Anthropometric distortion | IPD, philtrum, vertical thirds (MediaPipe 468-pt) |
| `run_illumination` | `illumination_tool.py` | 0.10 | Tier 1 | Lighting inconsistency | Gradient-based directional light analysis |
| `run_corneal` | `corneal_tool.py` | 0.10 | Tier 2 | Missing/mismatched catchlights | Bilateral reflection detection + divergence score |
| `run_rppg` | `rppg_tool.py` | 0.35* | Tier 2 | Absent biological signal | POS rPPG + FFT (0.7–4 Hz cardiac band) |

> *rPPG weight is only active in the CPU→GPU gate calculation. In the ensemble it uses `WEIGHT_RPPG` from thresholds.py. Video-only.

### GPU Tools (`.venv_gpu` — Sequential VRAM Loading)

| Tool | File | VRAM | Weight | Trust | Architecture | Checkpoint |
|---|---|---|---|---|---|---|
| `run_freqnet` | `freqnet_tool.py` | 0.4 GB | 0.09 | Tier 1 | CNNDetect ResNet-50 + FADHook DCT | [Wang et al. CVPR 2020](https://github.com/PeterWang4158/CNNDetect) |
| `run_univfd` | `univfd_tool.py` | 0.6 GB | 0.20 | Tier 3 | CLIP-ViT-L/14 + 4KB linear probe | [Ojha et al. CVPR 2023](https://github.com/ojha-group/UnivFD) |
| `run_xception` | `xception_tool.py` | 0.5 GB | 0.15 | Tier 2 | Xception (FaceForensics++) | [HongguLiu/Deepfake-Detection](https://github.com/HongguLiu/Deepfake-Detection) |
| `run_sbi` | `sbi_tool.py` | 0.8 GB | 0.20 | Tier 3 | EfficientNet-B4 + GradCAM | [mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages) |

> **Peak VRAM:** Never additive. Load → Run → `synchronize → del → gc → empty_cache` between each tool. Max single model = 0.8 GB (SBI).

---

## 🆕 Key Changes v3.0

### Bug Fixes Applied Since Last Commit

| # | File | Change | Impact |
|---|---|---|---|
| 1 | `core/agent.py` | `_safe_execute_tool` now wraps `tool.execute` in `ThreadPoolExecutor(timeout=30)` | CPU tools can no longer hang the pipeline |
| 2 | `core/agent.py` | GPU tools wrapped in `ThreadPoolExecutor(timeout=60)` with per-model VRAM requirements dict | GPU tools timeout properly; no hardcoded 0.6 GB |
| 3 | `core/agent.py` | Confidence aggregation changed to **directional**: `direction = (score - 0.5) × 2 × confidence × weight` | Gate uses directional confidence, not blind magnitude |
| 4 | `core/agent.py` | Unison check filters `score == 0.5` via `decisive_results` list | Neutral scores no longer miscounted as "REAL" |
| 5 | `utils/preprocessing.py` | `max_confidence` derived from `len(trajectory_bboxes) / len(frames)` | Face gate confidence dimension is now meaningful |
| 6 | `core/tools/rppg_tool.py` | `face_window = (0,0)` returns `ABSTAIN` and `continue` instead of processing all frames | rPPG no longer processes noise as signal |
| 7 | `core/agent.py` | `torch` + `vram_manager` + `ThreadPoolExecutor` all imported at top-level | No import-inside-loop overhead |
| 8 | `core/agent.py` | GPU lambda uses `make_loader(t)` / `make_inference(data)` factories | Closure captures tool by value; loop-refactor safe |
| 9 | `core/agent.py` | `is_degraded` propagated to `VERDICT` event AND return dict | Consumers can detect degraded analysis |
| 10 | `core/agent.py` | `_make_error_result(tool_name, msg, start_time)` centralizes all 4 error `ToolResult` constructors | Single source of truth for error shape |
| 11 | `utils/preprocessing.py` | `face_crop = frame[cy1:cy2, cx1:cx2]` (was `cx1:x2` typo) | Sharpness scoring uses fully clamped coordinates |

---

## 📁 File-by-File Reference

### `core/agent.py` — The Orchestrator

The nerve center. Implements `ForensicAgent` as a generator-based orchestrator:

```
ForensicAgent
├── __init__()              — Loads tool registry, ESC, EnsembleAggregator
├── _make_error_result()    — DRY factory for all ToolResult error states
├── _safe_execute_tool()    — CPU tool runner with 30s timeout via ThreadPoolExecutor
└── analyze()               — Main generator pipeline
    ├── Face Gate           — 4-dimension routing decision
    ├── Segment A: CPU      — Heuristic-gated tool execution + C2PA short-circuit
    ├── Segment B: Gate     — Directional conf + unison + domain check → HALT/MINIMAL/FULL
    ├── Segment C: GPU      — Closure-safe VRAM-managed sequential inference (60s timeout)
    ├── DEGRADED check      — flags >50% error rate
    └── Ensemble + LLM      — Final verdict with degraded flag
```

Key constants defined here:
```python
GPU_VRAM_REQUIREMENTS = {
    "run_freqnet": 0.4,   # GB
    "run_univfd":  0.6,
    "run_xception":0.5,
    "run_sbi":     0.8,
}

FACE_GATE_THRESHOLDS = {
    "min_confidence":         0.60,
    "min_face_area_ratio":    0.01,
    "min_frames_with_faces":  0.30,
}
```

---

### `utils/preprocessing.py` — Face Extraction & Tracking

Two-phase processing for video, one-phase for images:

```
PHASE 1 (video only) — Build Trajectories
  MediaPipe FaceMesh (static_image_mode=True per frame)
  CPU-SORT tracker (Kalman + Hungarian matching)
  → established_tracks: Dict[track_id, TrackedFace]
     └── trajectory_bboxes: Dict[frame_idx, (x1,y1,x2,y2)]

PHASE 2 — Extract Crops & Heuristics
  For each track with len(trajectory) ≥ min_track_length:
    face_window = longest contiguous run of frames
    max_confidence = len(trajectory) / total_frames   ← v3.0 fix
    Select sharpest frame (Laplacian variance)
    Generate face_crop_224, face_crop_380
    Extract 6 anatomical patches (periorbital, nasolabial, hairline, jaw)
    Set heuristic_flags: MOTION_BLUR, OCCLUSION, FACE_TOO_SMALL, LOW_LIGHT

PreprocessResult fields (relevant to gate):
  has_face                bool
  max_confidence          float  ← tracking coverage ratio [0–1]
  max_face_area_ratio     float  ← peak face/frame area ratio
  frames_with_faces_pct   float  ← % frames containing any face
  heuristic_flags         List[str]
  insufficient_temporal_data bool
```

**TrackedFace dataclass** (dict-compatible via `get()` and `__getitem__`):
```python
TrackedFace:
  identity_id       int
  landmarks         np.ndarray  (478, 2)
  trajectory_bboxes Dict[int, Tuple[int,int,int,int]]
  face_window       Tuple[int, int]   ← (start_frame, end_frame)
  face_crop_224     np.ndarray
  face_crop_380     np.ndarray
  patch_left_periorbital, patch_right_periorbital
  patch_nasolabial_left, patch_nasolabial_right
  patch_hairline_band, patch_chin_jaw
  heuristic_flags   List[str]
```

---

### `core/tools/rppg_tool.py` — Remote Photoplethysmography

Video-only physiological liveness detector:

```
_run_inference()
  ├── Skip if image (returns SKIPPED)
  ├── Skip if frames < RPPG_MIN_FRAMES (returns ABSTAIN)
  ├── Backup face check if tracked_faces empty (_lightweight_face_check)
  │
  └── For each tracked face:
        face_window check                      ← v3.0 fix
          (0,0) → append ABSTAIN, continue     ← was: process all frames as noise
        Slice frames[start:end]
        Remap trajectory keys to relative indices

        Extract POS signals for 3 ROIs:
          forehead, left_cheek, right_cheek
          Hair occlusion guard (Laplacian variance > 35.0 → ABSTAIN)
          Darkness guard (mean < 50 → return None)

        _evaluate_liveness():
          AMBIGUOUS     (hair occlusion)
          NO_PULSE      (flat signal across all ROIs)
          SYNTHETIC_FLATLINE (< 2 ROIs with variance)
          WEAK_PULSE_FAILED  (only 1 ROI passes)
          INCOHERENT    (peaks don't synchronize)
          PULSE_PRESENT (≥2 ROIs coherent within 0.05 Hz)

Best face by confidence wins. Returns ToolResult with liveness_label.
```

---

### `utils/ensemble.py` — Score Aggregation

```
calculate_ensemble_score()
  Step 1: Extract context (DCT peak ratio, compression flag)
  Step 2: C2PA override check (with visual corroboration guard)
  Step 3: Route each tool via _route()
    each tool returns (contribution, effective_weight)
    Routing rules:
      rPPG    → threshold gated (uses implied prob only at extremes)
      SBI     → blind spot below threshold; mid-band uses UnivFD context
      FreqNet → blind spot below threshold; compression discount
      UnivFD/Xception → direct score × weight
  Step 4: Suspicion Overdrive
    max_prob = max(gpu_specialist implied probs)
    if max_prob > SUSPICION_OVERRIDE_THRESHOLD:
        fake_score = max_prob   (hard max-pooling)
    else:
        fake_score = weighted_average

ensemble_score = 1.0 - fake_score   (REAL probability)
verdict = "FAKE" if ensemble_score ≤ ENSEMBLE_REAL_THRESHOLD else "REAL"
```

---

### `utils/vram_manager.py` — GPU Memory Lifecycle

```
VRAMLifecycleManager (context manager)
  __enter__:
    Acquire global RLock (120s timeout)
    Check VRAM availability (required_gb threshold)
    CPU fallback if insufficient
    Load model, move to device, set .eval()
  __exit__ / _safe_cleanup:
    model.to("cpu")
    torch.cuda.synchronize()
    del self.model
    _cleanup_device_memory() → empty_cache / mark_step (TPU)
    Release RLock

Hardware priority: TPU → CUDA → MPS → CPU

run_with_vram_cleanup(model_loader, inference_fn, model_name, required_vram_gb)
  → Wraps VRAMLifecycleManager + torch.no_grad()
  → Called from GPU phase in agent.py via ThreadPoolExecutor(timeout=60)
```

---

### `core/early_stopping.py` — Evidential Subjective Logic

The ESL controller is still instantiated but the v3.0 architecture primarily uses the **inline CPU→GPU gate** in `agent.py` for routing decisions. The ESC is available for future re-integration:

```
EarlyStoppingController.evaluate():
  1. C2PA hardware lock → immediate HALT
  2. Validate tool names against registry
  3. Compute weighted_sum, weights_run, weights_pending
  4. Evidential Subjective Logic:
     e_fake = Σ(weight_i × max(0, score_i - 0.5) × 2)
     e_real = Σ(weight_i × max(0, 0.5 - score_i) × 2)
     conflict_ratio = min(e_fake, e_real) / max(e_fake, e_real)
     if conflict_ratio > 0.35 → CONTINUE (adversarial conflict)
  5. Mathematical bounds:
     max_possible = (sum + 1.0 × pending_weight) / total_viable
     min_possible = (sum + 0.0 × pending_weight) / total_viable
  6. Locked FAKE check → HALT if min_possible > real_threshold
  7. Default → CONTINUE

Note: HALT_LOCKED_REAL is intentionally disabled — system always runs
GPU tools before declaring REAL to prevent sophisticated fake bypass.
```

---

## 🌐 Web Interface

The web UI is a premium glassmorphic dark-mode interface built with vanilla HTML/CSS/JS and served by FastAPI.

### Features

- **Drag & Drop Upload** — Accepts JPEG, PNG, WebP, and video files
- **Real-Time SSE Streaming** — Watch each tool execute live with progress indicators
- **GATE_DECISION Event** — UI shows whether GPU was triggered, minimized, or halted
- **DEGRADED Banner** — Displayed when `degraded: true` in VERDICT event
- **Verdict Banner** — Color-coded AUTHENTIC (green) / TAMPERED (red) with composite confidence score
- **Tool Cards** — Score, confidence, and evidence summary per forensic tool
- **Agent Synthesis Panel** — LLM-generated natural language forensic explanation

### SSE Event Stream

| Event | When | Payload |
|---|---|---|
| `PIPELINE_SELECTED` | After face gate | `{face_pipeline: bool}` |
| `TOOL_STARTED` | Before each tool | `{tool_name}` |
| `TOOL_COMPLETED` | After each tool | `{success, confidence}` |
| `EARLY_STOP` | C2PA verified | `{reason: "C2PA_VERIFIED"}` |
| `GATE_DECISION` | After CPU phase | `{decision, confidence, unison}` |
| `llm_start` | Before LLM | — |
| `VERDICT` | Final | `{verdict, score, explanation, degraded}` |

### Scoring Convention

| Score | Convention | Meaning |
|---|---|---|
| `0.0` | AUTHENTIC | Tool found no manipulations |
| `0.5` | NEUTRAL / ERROR | Tool abstained or errored |
| `1.0` | TAMPERED | Tool detected manipulation artifacts |

> **Critical:** `ensemble_score` is the **REAL probability** (1.0 = authentic, 0.0 = fake). The individual tool `score` field is the **FAKE probability**.

---

## ⚙️ Configuration

### Environment Variables

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `AEGIS_MODEL_DIR` | `models/` | Root directory for all model weight files |
| `AEGIS_DEVICE` | `auto` | Force `cuda`, `cpu`, or `auto` detect |
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama LLM server URL |
| `OLLAMA_MODEL` | `phi3:mini` | LLM model for verdict synthesis |
| `AEGIS_VRAM_THRESHOLD` | `3.5` | Minimum free VRAM (GB) to attempt GPU tools |
| `LLM_TEMPERATURE` | `0.1` | Low temperature for deterministic forensic output |
| `LLM_MAX_TOKENS` | `1024` | Maximum tokens for LLM response |

### Thresholds (`utils/thresholds.py` — Single Source of Truth)

| Threshold | Value | Purpose |
|---|---|---|
| `ENSEMBLE_REAL_THRESHOLD` | 0.15 | `ensemble_score` ≤ this → FAKE verdict |
| `ENSEMBLE_FAKE_THRESHOLD` | 0.85 | Score ≥ this → confident FAKE |
| `SUSPICION_OVERRIDE_THRESHOLD` | 0.75 | Hard max-pooling trigger for GPU specialists |
| `CONFLICT_STD_THRESHOLD` | 0.20 | Tool disagreement level flagged as conflict |
| `SBI_FAKE_THRESHOLD` | 0.60 | SBI score above this = blend detected |
| `UNIVFD_FAKE_THRESHOLD` | 0.60 | UnivFD score above this = generative AI detected |
| `RPPG_CARDIAC_BAND_MIN_HZ` | 0.7 | Lower bound of cardiac frequency (42 BPM) |
| `RPPG_CARDIAC_BAND_MAX_HZ` | 4.0 | Upper bound of cardiac frequency (240 BPM) |
| `RPPG_HAIR_OCCLUSION_VARIANCE` | 35.0 | Laplacian variance threshold for hair occlusion |

---

## 📁 Project Structure

```
aegis-x/
├── run_web.py                  # Entry point — FastAPI + SSE streaming server
├── setup.py                    # One-click installer (venvs + Kaggle weights)
├── verify_tools.py             # Diagnostic: test all tools individually
│
├── core/
│   ├── agent.py                # ★ ForensicAgent — Full dual-pipeline orchestration
│   │                           #   Face Gate → CPU → CPU/GPU Gate → GPU → Ensemble → LLM
│   ├── early_stopping.py       # Evidential Subjective Logic gating (ESC)
│   ├── forensic_summary.py     # LLM prompt builder with grounded evidence
│   ├── llm.py                  # Ollama HTTP client bridge
│   ├── memory.py               # SQLite-backed case memory system
│   ├── config.py               # Typed dataclass configuration hierarchy
│   ├── data_types.py           # ToolResult contract (score/fake_score alias)
│   ├── base_tool.py            # Abstract base class for all forensic tools
│   ├── subprocess_proxy.py     # Bridge: .venv_main ↔ .venv_gpu
│   ├── subprocess_worker.py    # GPU worker process (runs in .venv_gpu)
│   ├── exceptions.py           # Custom exception hierarchy
│   └── tools/
│       ├── registry.py         # Tool manifest, weights, trust tiers, dispatch
│       ├── c2pa_tool.py        # [CPU] C2PA cryptographic provenance check
│       ├── dct_tool.py         # [CPU] JPEG double-quantization detection
│       ├── geometry_tool.py    # [CPU] 468-landmark anthropometric ratio analysis
│       ├── illumination_tool.py# [CPU] Directional lighting gradient analysis
│       ├── corneal_tool.py     # [CPU] Catchlight reflection divergence scoring
│       ├── rppg_tool.py        # [CPU] POS rPPG + FFT cardiac liveness (video)
│       ├── univfd_tool.py      # [GPU] CLIP-ViT-L/14 + linear probe (CVPR 2023)
│       ├── xception_tool.py    # [GPU] XceptionNet (FaceForensics++)
│       ├── sbi_tool.py         # [GPU] Self-Blended Images + GradCAM
│       ├── freqnet_tool.py     # [GPU] CNNDetect + FADHook frequency fusion
│       └── freqnet/
│           ├── preprocessor.py # DCT + spatial preprocessing
│           ├── fad_hook.py     # Frequency Artifact Detection hooks
│           └── calibration.py  # Z-score baseline calibration
│
├── utils/
│   ├── vram_manager.py         # GPU lifecycle (sync→del→gc→empty_cache) + RLock
│   ├── preprocessing.py        # ★ MediaPipe + CPU-SORT tracking + face crops
│   ├── ensemble.py             # Weighted scoring + Suspicion Overdrive + C2PA guard
│   ├── thresholds.py           # ★ Central numeric constants (single source of truth)
│   ├── ollama_client.py        # HTTP client for local Ollama server
│   ├── video.py                # Video I/O (torchcodec → OpenCV fallback)
│   ├── image.py                # Image I/O helpers
│   └── logger.py               # Structured logging setup
│
├── web/
│   ├── index.html              # Glassmorphic dark-mode UI
│   ├── style.css               # Full design system (Outfit font, animations)
│   └── script.js               # SSE client, tool card rendering, verdict display
│
├── models/                     # .gitignored — downloaded via setup.py
│   ├── clip-vit-large-patch14/ # CLIP ViT-L/14 backbone (~890 MB)
│   ├── univfd/probe.pth        # Linear probe (4 KB)
│   ├── xception/xception_deepfake.pth  # XceptionNet (80 MB)
│   ├── sbi/efficientnet_b4.pth         # EfficientNet-B4 (135 MB)
│   └── freqnet/cnndetect_resnet50.pth  # CNNDetect ResNet-50 (270 MB)
│
├── requirements-main.txt       # Main env: FastAPI, MediaPipe, OpenCV, scipy
├── requirements-gpu.txt        # GPU env: PyTorch, Transformers, timm
├── .env.example                # Environment variable template
└── .gitignore
```

> ★ = Most heavily modified files in v3.0

---

## 🧪 Diagnostics

```bash
# Test with a specific image
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv_main/bin/python verify_tools.py --image path/to/face.jpg

# Test with default image
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv_main/bin/python verify_tools.py
```

---

## 📊 Benchmarking Datasets

| Dataset | Year | Threat Class | Primary Tools |
|---|---|---|---|
| [GenImage](https://github.com/GenImage-Dataset/GenImage) | 2023 | Text-to-image AI (Midjourney, SD, DALL-E) | `run_univfd` |
| [ArtiFact](https://github.com/awsaf49/artifact) | 2023 | Multi-generator + real-world compression | `run_univfd`, `run_freqnet` |
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | 2019 | Face-swap and reenactment | `run_xception`, `run_sbi` |
| [ForgeryNet](https://github.com/yinanhe/forgerynet) | 2021 | Surgical face edits, morphs | `run_sbi` |
| [WildDeepfake](https://github.com/deepfakeinthewild/deepfake-in-the-wild) | 2020 | Internet-scraped deepfakes | Full pipeline |
| [DiffusionForensics](https://github.com/ZhendongWang6/DIRE) | 2023 | Diffusion outputs with heavy compression | `run_freqnet` |

---

## 🔑 Model Weights

**Dataset:** [kaggle.com/datasets/gauravkumarjangid/aegis-pth](https://www.kaggle.com/datasets/gauravkumarjangid/aegis-pth)

| Weight File | Size | Model | Source Paper |
|---|---|---|---|
| `probe.pth` | 4 KB | UnivFD linear probe | Ojha et al., *Towards Universal Fake Image Detectors*, CVPR 2023 |
| `xception_deepfake.pth` | 80 MB | Xception (FaceForensics++) | Rössler et al., *FaceForensics++*, ICCV 2019 |
| `efficientnet_b4.pth` | 135 MB | SBI EfficientNet-B4 | Shiohara & Yamasaki, *Detecting Deepfakes with Self-Blended Images*, CVPR 2022 |
| `cnndetect_resnet50.pth` | 270 MB | CNNDetect ResNet-50 | Wang et al., *CNN-generated images are surprisingly easy to spot*, CVPR 2020 |

CLIP-ViT-L/14 (~890 MB) auto-downloads from HuggingFace on first run.

---

## 📚 References

```bibtex
@inproceedings{ojha2023universal,
  title={Towards Universal Fake Image Detectors that Generalize Across Generative Models},
  author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
  booktitle={CVPR}, year={2023}
}

@inproceedings{shiohara2022sbi,
  title={Detecting Deepfakes with Self-Blended Images},
  author={Shiohara, Kaede and Yamasaki, Toshihiko},
  booktitle={CVPR}, year={2022}
}

@inproceedings{rossler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={R{\"o}ssler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={ICCV}, year={2019}
}

@inproceedings{wang2020cnndetect,
  title={CNN-generated images are surprisingly easy to spot... for now},
  author={Wang, Sheng-Yu and Wang, Oliver and Zhang, Richard and Owens, Andrew and Efros, Alexei A},
  booktitle={CVPR}, year={2020}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with 🔬 forensic rigor and ⚡ engineering precision.<br>
  <em>v3.0 — Dual-Pipeline Architecture with Face Gate, Directional Confidence Gating & Silent Killer Fixes</em>
</p>
