# AEGIS-X IMPLEMENTATION PLAN: PART 3

## PHASE 3: GPU Forensic Tools (Days 11–14)

*(Note: All GPU tools MUST utilize the `VRAMLifecycleManager` defined in Day 5 to prevent OOM errors on 4GB hardware.)*

### Day 11: Universal Forgery Tool (CLIP + Adapter)

#### Prompt for Day 11:

**Section A: Context Reminder**
Aegis-X is an offline, agentic multi-modal forensic engine. 
This is Phase 3, Day 11. We are building the first GPU tool: The CLIP-based Universal Forgery Detector. Because fully-synthetic faces (e.g., from Sora or Midjourney) lack the "blend boundaries" of face-swaps, we rely on CLIP's broad representation (learned from 400M real images) paired with a lightweight forensic adapter.

**Section B: Today's Objectives**
- Create `core/tools/clip_adapter_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `ClipAdapterTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_clip_adapter"`
  - `def setup(self)`: No-op. The actual model loading happens dynamically in inference to save VRAM.
  - `def _load_model(self) -> torch.nn.Module`:
    - Imports `clip` and loads `clip.load("ViT-B/32", device="cpu")`.
    - Freezes CLIP backbone (`requires_grad = False`).
    - Initializes the Adapter: a 2-layer MLP `(512 -> 256 -> 1)` with `ReLU` and `Sigmoid`. 
    - Loads adapter weights from `AegisConfig().models.clip_adapter_weights`. If weights are missing, set a fallback flag.
    - Return the combined module.
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - Expected input: `input_data["patch_left_eye"]`, `"patch_right_eye"`, `"patch_hairline"`, `"patch_jaw"`.
    - If any patch is missing, return neutral result `score=0.5`.
    - Wrap the inference block using: `with VRAMLifecycleManager(self._load_model) as model:`
    - For each 224x224 patch: Normalize using CLIP's standard `(0.48145466, 0.4578275, 0.40821073)` mean and `(0.26862954, 0.26130258, 0.27577711)` std.
    - Pass through CLIP to get 512-dim embedding, then pass through the MLP adapter to get a score between 0 and 1.
    - Aggregate scores across patches: `final_score = (max_score * 0.6) + (mean_score * 0.4)`.
    - Determine which patch caused the highest anomaly to report it.
    - **Fallback logic:** If ML adapter weights were missing, compute Zero-Shot cosine similarity between the patch's `image_features` and the text embeddings for two phrases: "a real photograph of a face" vs "a digitally manipulated or AI-generated face". Use softmax to get the probability of fake.
    - `confidence`: `abs(final_score - 0.5) * 2.0` (Closer to 0 or 1 = higher confidence).
    - `evidence_summary`: "CLIP Adapter found severe anomalies in the [Patch Name] region" or "CLIP Adapter found no signs of universal synthesis."

**Section D: Implementation Rules for That Day**
- Import `VRAMLifecycleManager` from `utils.vram_manager`.
- Do not keep the CLIP model instantiated on the class level.

**Section E: Testing & Verification Steps**
Create `test_day11.py`:
```python
import numpy as np
import torch
from core.tools.clip_adapter_tool import ClipAdapterTool
from core.config import AegisConfig

def test_day11():
    tool = ClipAdapterTool()
    
    # Create dummy patches
    patches = {
        "patch_left_eye": np.zeros((224, 224, 3), dtype=np.uint8),
        "patch_right_eye": np.zeros((224, 224, 3), dtype=np.uint8),
        "patch_hairline": np.zeros((224, 224, 3), dtype=np.uint8),
        "patch_jaw": np.zeros((224, 224, 3), dtype=np.uint8),
    }
    
    result = tool.execute(patches)
    print(f"CLIP Adapter logic evaluated: Score {result.score}, Success: {result.success}")
    assert result.success is True
    assert 0.0 <= result.score <= 1.0
    
    # Verify memory is flushed
    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() == 0
    print("✅ Day 11 CLIP tool tested. VRAM safely relinquished.")

if __name__ == "__main__":
    test_day11()
```
Run `python test_day11.py`.
Expected output: Evaluates dummy data via zero-shot (or adapter if weights supplied), completes without OOM, returns valid `ToolResult`.

**Section F: Files Produced**
- `core/tools/clip_adapter_tool.py`
- Depends on: `clip` library, `VRAMLifecycleManager`.
- Enables: High-confidence detection of fully synthetic diffusions and GANs.

#### Day 11 Summary:
- Files: core/tools/clip_adapter_tool.py
- Depends on: VRAMLifecycleManager
- Enables: Universal forgery detection.

---

### Day 12: Blend Boundary Tool (SBI)

#### Prompt for Day 12:

**Section A: Context Reminder**
Aegis-X is a forensic engine. 
This is Phase 3, Day 12. We are implementing the SBI (Self-Blended Images) detector. It utilizes an EfficientNet-B4 backbone to specifically detect face-swap "blend boundaries". The model is trained to recognize the seam where a source face is pasted onto a target frame, making it generator-agnostic for face-swaps.

**Section B: Today's Objectives**
- Create `core/tools/sbi_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `SBITool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_sbi"`
  - `def _load_model(self) -> torch.nn.Module`:
    - Imports `torchvision.models.efficientnet_b4`.
    - Replaces the classifier head to output a single value (`in_features=1792`, `out_features=1`).
    - Loads weights from `AegisConfig().models.sbi_weights`. (If missing, default to neutral random initialized for testing, but log warning).
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - Expected input: `input_data["face_crop_380"]`. If missing, return `score=0.5`.
    - **CRITICAL**: Conditional Skip Logic! The agent will provide `input_data["clip_score"]`. If `clip_score > 0.70`, it implies the face is fully synthetic. SBI only detects swap boundaries and will fail on fully synthetic faces. IF `clip_score > 0.70`, immediately return `success=True, score=0.5, confidence=0.0, evidence_summary="Skipped SBI: High CLIP score indicates fully synthetic face, not a face-swap."`
    - Wrap inference: `with VRAMLifecycleManager(self._load_model) as model:`
    - Convert `uint8` [380, 380, 3] image to float tensor `[1, 3, 380, 380]`, normalize with ImageNet means `[0.485, 0.456, 0.406]` and stds `[0.229, 0.224, 0.225]`.
    - Pass through model, apply Sigmoid.
    - Result logic:
      - `score`: the direct sigmoid output `[0, 1]`.
      - `confidence`: `min(0.85, abs(score - 0.5) * 2.0)`
      - `evidence_summary`: "SBI EfficientNet detected blending boundaries indicative of face-swap operations." OR "No blending boundaries found."

**Section D: Implementation Rules for That Day**
- Do not forget the conditional skip based on the `clip_score`. This is a critical piece of the 'agent' architecture.
- EfficientNet-B4 expects [380, 380] resolution exactly.

**Section E: Testing & Verification Steps**
Create `test_day12.py`:
```python
import numpy as np
import torch
from core.tools.sbi_tool import SBITool

def test_day12():
    tool = SBITool()
    dummy_input = {"face_crop_380": np.zeros((380, 380, 3), dtype=np.uint8)}
    
    # 1. Test Skip Logic
    skip_input = {**dummy_input, "clip_score": 0.85}
    res_skip = tool.execute(skip_input)
    assert res_skip.score == 0.5
    assert "Skipped SBI" in res_skip.evidence_summary
    print("✅ Day 12 SBI Skip logic works!")
    
    # 2. Test Execution
    run_input = {**dummy_input, "clip_score": 0.4}
    res_run = tool.execute(run_input)
    assert res_run.success is True
    
    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() == 0
    print("✅ Day 12 SBI execution + VRAM flush successful!")

if __name__ == "__main__":
    test_day12()
```
Run `python test_day12.py`.
Expected output: Skip logic bypassed the model successfully, while standard logic evaluated the tensor and purged VRAM.

**Section F: Files Produced**
- `core/tools/sbi_tool.py`
- Depends on: `torchvision`, `VRAMLifecycleManager`
- Enables: Face-swap border detection.

#### Day 12 Summary:
- Files: core/tools/sbi_tool.py
- Depends on: torchvision
- Enables: Blending artifact tracking.

---

### Day 13: Frequency Neural Tool (FreqNet)

#### Prompt for Day 13:

**Section A: Context Reminder**
This is Phase 3, Day 13 of the Aegis-X build. We are implementing the final GPU tool: FreqNet (F3Net ResNet-50). While our DCT tool calculates hard mathematical frequency histograms, FreqNet uses a parallel ResNet architecture to learn abstract frequency-domain anomalies common to GANs.

**Section B: Today's Objectives**
- Create `core/tools/freqnet_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `FreqNetTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_freqnet"`
  - `def _load_model(self) -> torch.nn.Module`:
    - Imports `torchvision.models.resnet50`.
    - Change fully connected layer to `out_features=1`.
    - Load weights from `AegisConfig().models.freqnet_weights`. (If missing, fallback to warning block).
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - Expected input: `input_data["face_crop_224"]` (since we focus just on the face frequency).
    - If missing, return `score=0.5`.
    - Wrap inference: `with VRAMLifecycleManager(self._load_model) as model:`
    - The actual F3Net paper uses a dual-stream DCT branch. For this implementation to fit standard weights, assume the provided weights are a standard ResNet-50 fine-tuned on the high-pass filtered DCT domain of deepfakes.
    - Pre-processing step before inputting to the network: applies a 2D DCT to the image (using `scipy`), applies a high-pass filter (zero out the central low-frequency 32x32 block), applies inverse DCT (`idct`), then passes THAT filtered image to the ResNet.
    - Image normalization: ImageNet means and stds.
    - Pass through network -> Sigmoid -> `score`.
    - `confidence`: `min(0.85, abs(score - 0.5) * 2.0)`
    - `evidence_summary`: "Neural frequency analysis (FreqNet) detected high-frequency GAN artifacts." OR "High-frequency spectrum matches natural camera sensor noise."

**Section D: Implementation Rules for That Day**
- The high-pass filter is crucial: PyTorch's ResNet must ingest the *residual/high-frequency* elements, not the base RGB.

**Section E: Testing & Verification Steps**
Create `test_day13.py`:
```python
import numpy as np
import torch
from core.tools.freqnet_tool import FreqNetTool

def test_day13():
    tool = FreqNetTool()
    dummy_input = {"face_crop_224": np.ones((224, 224, 3), dtype=np.uint8) * 128}
    
    res = tool.execute(dummy_input)
    assert res.success is True
    
    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() == 0
    print("✅ Day 13 FreqNet execution + VRAM flush successful!")

if __name__ == "__main__":
    test_day13()
```
Run `python test_day13.py`.
Expected output: FreqNet filter evaluates and returns results without OOM.

**Section F: Files Produced**
- `core/tools/freqnet_tool.py`
- Depends on: `torchvision`, `scipy.fft`
- Enables: Advanced GAN artifact detection.

#### Day 13 Summary:
- Files: core/tools/freqnet_tool.py
- Depends on: torchvision.models
- Enables: Coverage against GAN fingerprints.

---

### Day 14: Tool Registry

#### Prompt for Day 14:

**Section A: Context Reminder**
Aegis-X is an agentic engine. The agent doesn't blindly loop files; it dynamically calls tools by name from a registry.
This is Phase 3, Day 14. All 8 forensic tools are written. Today we group them into a singleton registry for the agent to query and invoke.

**Section B: Today's Objectives**
- Create `core/tools/tool_registry.py` and populate it with instances of all 8 tools.

**Section C: Detailed Specifications**
- `class ToolRegistry`:
  - `def __init__(self)`:
    - Initialize an internal `self.tools: dict[str, BaseForensicTool] = {}`.
    - Import and instantiate all 8 tools (C2PA, RPPG, DCT, Geometry, Illumination, ClipAdapter, SBI, FreqNet).
    - Register them by their `tool_name` property.
  - `def execute_tool(self, name: str, input_data: dict) -> ToolResult`:
    - Lookup the tool.
    - If found, call `tool.execute(input_data)`.
    - If not found, return a `ToolResult` marked `success=False`, `error=Tool not found`.
  - `def get_tool_names(self) -> list[str]`: Return a list of all registered tool names.

**Section D: Implementation Rules for That Day**
- Handle potential circular imports gracefully.
- Ensure the registry acts as a factory/singleton so tools don't need to be re-instantiated constantly.

**Section E: Testing & Verification Steps**
Create `test_day14.py`:
```python
from core.tools.tool_registry import ToolRegistry

def test_day14():
    registry = ToolRegistry()
    names = registry.get_tool_names()
    print(f"Registered Tools: {names}")
    
    assert len(names) == 8
    assert "run_geometry" in names
    assert "check_c2pa" in names
    assert "run_clip_adapter" in names
    
    # Test invalid lookup
    err_res = registry.execute_tool("fake_tool", {})
    assert err_res.success is False
    assert "not found" in err_res.error
    
    print("✅ Day 14 Tool Registry functional and populated with all 8 tools.")

if __name__ == "__main__":
    test_day14()
```
Run `python test_day14.py`.
Expected output: Successfully imports every tool without error and shows an 8-item roster.

**Section F: Files Produced**
- `core/tools/tool_registry.py`
- Depends on: All `core/tools/*` implementations.
- Enables: The agent orchestrator can cleanly request tool executions.

#### Day 14 Summary:
- Files: core/tools/tool_registry.py
- Depends on: All Tool modules
- Enables: Unified point of access for Phase 5 LLM Agent.

---

## PHASE 4: Ensemble, Early Stopping & Memory (Days 15–17)

### Day 15: Weighted Ensemble Scorer

#### Prompt for Day 15:

**Section A: Context Reminder**
Aegis-X uses a weighted ensemble of orthogonal tools to defeat deepfakes. The ensemble score is not a simple average, but rather a confidence-weighted normalized aggregation.
This is Phase 4, Day 15. We are building the mathematics required to aggregate `ToolResult`s into a single 0-1 probability.

**Section B: Today's Objectives**
- Create `utils/ensemble.py`.

**Section C: Detailed Specifications**
- Variables: Import exact base weights from `core.config.AegisConfig().weights`.
  (Base: CLIP=0.30, SBI=0.20, FreqNet=0.20, rPPG=0.15, DCT=0.10, Geo=0.03, Illum=0.02)
- `def calculate_ensemble_score(tool_results: list[ToolResult]) -> dict`:
  - Input: A list of executed tool results.
  - Return: A dict containing `{"ensemble_score": float, "is_c2pa_override": bool}`.
  - Step 1: Check C2PA. If the `check_c2pa` tool returned `score == 0.0` and `confidence == 1.0` (Signed), return `ensemble_score=0.0` and `is_c2pa_override=True` immediately.
  - Step 2: For each tool, `effective_weight_i = base_weight_i * confidence_i`.
  - Step 3: Compute total effective weight sum.
  - Step 4: Normalize weights: `norm_weight_i = effective_weight_i / total_effective_weights` (Handle div by 0 case if all confidences are 0, return `score=0.5`).
  - Step 5: `ensemble_score = Sum(score_i * norm_weight_i)`.
  - Step 6: Calibration via Agreement. Calculate the standard deviation (`np.std`) of all valid tool scores. Let `std_dev = np.std(scores)`.
    - If `std_dev < 0.1` (high agreement), push the `ensemble_score` slightly towards the nearest extreme (0 or 1) by 5%.
    - If `std_dev > 0.3` (high conflict), push the `ensemble_score` slightly towards 0.5 (neutral) by 10%.
  - Wrap output strictly between 0.0 and 1.0.

**Section D: Implementation Rules for That Day**
- Ignore tools where `success=False` or `confidence == 0.0` from the normalization calculation completely.

**Section E: Testing & Verification Steps**
Create `test_day15.py`:
```python
from core.data_types import ToolResult
from utils.ensemble import calculate_ensemble_score

def test_day15():
    res1 = ToolResult(tool_name="run_clip_adapter", success=True, score=0.9, confidence=0.9, details={}, execution_time_ms=0, evidence_summary="")
    res2 = ToolResult(tool_name="run_geometry", success=True, score=0.8, confidence=0.8, details={}, execution_time_ms=0, evidence_summary="")
    
    # 1. Base test
    out = calculate_ensemble_score([res1, res2])
    print(f"Aggregated Score (High agreement): {out['ensemble_score']:.3f}")
    assert out["ensemble_score"] > 0.8
    assert not out["is_c2pa_override"]
    
    # 2. C2PA Override test
    c2pa = ToolResult(tool_name="check_c2pa", success=True, score=0.0, confidence=1.0, details={}, execution_time_ms=0, evidence_summary="")
    out_override = calculate_ensemble_score([res1, res2, c2pa])
    print(f"Override Score: {out_override['ensemble_score']:.3f}")
    assert out_override["ensemble_score"] == 0.0
    assert out_override["is_c2pa_override"]
    
    print("✅ Day 15 Ensemble aggregation math passed.")

if __name__ == "__main__":
    test_day15()
```
Run `python test_day15.py`.
Expected output: Evaluates custom weighting rules and respects the cryptographical override.

**Section F: Files Produced**
- `utils/ensemble.py`
- Depends on: `AegisConfig`
- Enables: Final probability resolution.

#### Day 15 Summary:
- Files: utils/ensemble.py
- Depends on: ToolResult
- Enables: Core mathematical decision making.

---

### Day 16: Early Stopping Controller

#### Prompt for Day 16:

**Section A: Context Reminder**
A traditional pipeline executes all steps. Aegis-X runs as an agent and respects an Early Stopping check to save compute time (up to 40-80% savings) on overwhelmingly obvious cases.
This is Phase 4, Day 16. We are building the logic that evaluates if the current ensemble score is locked down enough to skip the remaining heavy GPU tools.

**Section B: Today's Objectives**
- Create `core/early_stopping.py`.

**Section C: Detailed Specifications**
- `class EarlyStoppingController`:
  - `def __init__(self, thresholds: ThresholdConfig)`
  - `def evaluate(self, current_ensemble_score: float, tools_run: list[str], tools_pending: list[str]) -> bool:`
    - Uses threshold bounds (`0.15` and `0.85` by default).
    - Condition 1 (C2PA Signed): If `check_c2pa` was run and `current_ensemble_score == 0.0` (override triggered), STOP = True.
    - Condition 2: If `current_ensemble_score > 0.85`, STOP = True. (Locked fake).
    - Condition 3: If `current_ensemble_score < 0.15`, STOP = True. (Locked real).
    - Condition 4 (Diminishing potential): Calculate the theoretical maximum remaining weight. If the pending tools (e.g., Geometry, Illumination) possess too little weight to possibly pull the current score out of the "verdict threshold zone", STOP = True.
  - Return `True` to halt analysis, `False` to continue executing tools.

**Section D: Implementation Rules for That Day**
- Document the diminishing potential formula clearly. To calculate it, use the fixed base weights from `AegisConfig`.

**Section E: Testing & Verification Steps**
Create `test_day16.py`:
```python
from core.early_stopping import EarlyStoppingController
from core.config import AegisConfig

def test_day16():
    esc = EarlyStoppingController(AegisConfig().thresholds)
    
    # Very high score, should trigger early stop
    stop1 = esc.evaluate(0.92, ["run_clip_adapter", "run_freqnet"], ["run_sbi"])
    assert stop1 is True
    
    # Ambiguous score, should NOT stop
    stop2 = esc.evaluate(0.55, ["run_geometry"], ["run_clip_adapter"])
    assert stop2 is False

    print("✅ Day 16 Early Stopping controller obeys threshold rules!")

if __name__ == "__main__":
    test_day16()
```
Run `python test_day16.py`.
Expected output: Boolean results matching expectation.

**Section F: Files Produced**
- `core/early_stopping.py`
- Depends on: `AegisConfig`
- Enables: The agent loop to break early and yield verdicts 2-4x faster.

#### Day 16 Summary:
- Files: core/early_stopping.py
- Depends on: ThresholdConfig
- Enables: Massive compute saving optimization.

---

### Day 17: SQLite Memory System

#### Prompt for Day 17:

**Section A: Context Reminder**
Aegis-X doesn't just evaluate in a vacuum — it has a persistent memory. If an analyst provides feedback ("This was actually a false positive"), the system stores the vector signature of that file and avoids repeating the mistake.
This is Phase 4, Day 17. We are scaffolding the `SQLite`-backed memory system for experience learning.

**Section B: Today's Objectives**
- Create `core/memory.py`.

**Section C: Detailed Specifications**
- `class MemorySystem`:
  - `def __init__(self, db_path="data/memory.db")`: 
    Ensure directory `data/` exists. Initialize SQLite connection.
    Create table `cases`: `id (INTEGER PK), timestamp (TEXT), file_hash (TEXT UNIQUE), file_type (TEXT), verdict (TEXT), confidence (REAL), ensemble_score (REAL), tool_scores (TEXT/JSON), reasoning (TEXT), feedback_label (TEXT)`.
  - `def store_case(self, file_hash: str, file_type: str, verdict: str, confidence: float, ensemble: float, tool_scores_dict: dict, reasoning: str)`:
    Inserts or updates the record via UPSERT (`ON CONFLICT(file_hash) DO UPDATE`). Format the `tool_scores_dict` to a JSON string.
  - `def store_feedback(self, file_hash: str, actual_label: str)`:
    Updates `feedback_label` with user correction (e.g., "REAL" or "FAKE").
  - `def query_similar_history(self, current_tool_scores: dict) -> list[dict]`:
    Retrieve past cases. Calculate Euclidean distance between `current_tool_scores` vectors and historical `tool_scores_JSON` vectors. Return the top 3 closest historical evaluations where a `feedback_label` exists. This allows the LLM to write: "This exact combination of high CLIP but low FreqNet previously resulted in a False Positive."

**Section D: Implementation Rules for That Day**
- Keep queries lightweight. Use Python's built-in `sqlite3` and `json`.
- The euclidean distance logic can be executed by extracting rows into memory if the db is small, avoiding complex SQLite C-extensions.

**Section E: Testing & Verification Steps**
Create `test_day17.py`:
```python
import os
from core.memory import MemorySystem

def test_day17():
    os.makedirs("test_data", exist_ok=True)
    db_path = "test_data/test_memory.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    mem = MemorySystem(db_path)
    
    mock_tools = {"run_clip_adapter": 0.9, "run_rppg": 0.2}
    mem.store_case("abc123hash", "image", "FAKE", 0.85, 0.85, mock_tools, "Looks synthetic")
    mem.store_feedback("abc123hash", "REAL") # User corrects to REAL (False Positive scenario)
    
    mock_current = {"run_clip_adapter": 0.88, "run_rppg": 0.25}
    matches = mem.query_similar_history(mock_current)
    
    assert len(matches) == 1
    assert matches[0]["feedback_label"] == "REAL"
    print("✅ Day 17 SQLite memory stores cases, accepts feedback, and returns euclidean proximity matches!")
    
if __name__ == "__main__":
    test_day17()
```
Run `python test_day17.py`.
Expected output: The SQLite DB creates successfully, accepts operations, retrieves records, and calculates distance accurately.

**Section F: Files Produced**
- `core/memory.py`
- Depends on: `sqlite3`
- Enables: State persistence and evolutionary context across multiple runs.

#### Day 17 Summary:
- Files: core/memory.py
- Depends on: sqlite3
- Enables: LLM to reference historical fixes.

---
END OF PART 3. Say 'continue' for Part 4: Phase 5 (Days 18–22).
