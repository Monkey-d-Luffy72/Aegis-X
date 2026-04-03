# Aegis-X Dual-Pipeline Architecture Design

## Overview
Aegis-X leverages a dynamic dual-pipeline architecture (Face Pipeline vs. No-Face Pipeline) dynamically orchestrated by the `ForensicAgent`.

## Pipeline Segment A: CPU Phase
The CPU Phase executes immediately after preprocessing. It aims to rapidly identify clear signs of authenticity or structural synthetic instability using minimal resources.

- **Tools Run:**
  - `check_c2pa`: Validates cryptographic signatures. If successful, stops early and guarantees authenticity.
  - `run_dct`: Always runs to measure uniform compression.
  - **If Face Detected:**
    - `run_rppg`: Runs on the `face_window` (contiguous frame bounds) to extract biological signals unless `INSUFFICIENT_TEMPORAL_DATA` is flagged.
    - `run_geometry`, `run_illumination`, `run_corneal`: Configured dynamically using preprocessing heuristics fallbacks (`FACE_TOO_SMALL`, `LOW_LIGHT`, `MOTION_BLUR`, `OCCLUSION`).

## Pipeline Segment B: CPU->GPU Gate
After the CPU Phase, the system dynamically routes execution via Unison and Confidence Aggregation Rules to manage VRAM utilization optimally:

- **HALT**: (CPU Confidence > 0.93 AND Unison Agreement across domains). System skips GPU tools completely.
- **MINIMAL_GPU**: (CPU Confidence 0.80-0.93). Runs `run_univfd` only.
- **FULL_GPU**: (CPU Confidence < 0.80 or Unison disagreement). Entire GPU stack runs.

## Pipeline Segment C: GPU Phase
The GPU Phase processes deep-learning features. All network executions utilize strict `run_with_vram_cleanup` execution blocks sequentially.

- **For No-Face Media**: `run_freqnet` -> `run_univfd` -> `run_xception`
- **For Face Media**: `run_freqnet` -> `run_univfd` -> `run_xception` -> `run_sbi`

## Safety Routing
- `run_rppg` is guarded by dual safety mechanisms checking video bounds and fallback MediaPipe validation if upstream preprocessing falters.
- Exceptions explicitly route as `ERROR` and degrade the ensemble confidence safely.
