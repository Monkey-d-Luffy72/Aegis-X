# AEGIS-X IMPLEMENTATION PLAN

## Overview Table
| Phase | Name | Days | Key Deliverables |
|---|---|---|---|
| 0 | Project Scaffolding & Configuration | 1–2 | Repository structure, config.py, custom exceptions, logger setup, ToolResult dataclass, abstract base tool |
| 1 | Utilities & Preprocessing Pipeline | 3–5 | image.py, video.py, preprocessing.py (dlib face/landmark extraction), vram_manager.py |
| 2 | CPU Forensic Tools | 6–10 | tools/c2pa_tool.py, rppg_tool.py, dct_tool.py, geometry_tool.py, illumination_tool.py |
| 3 | GPU Forensic Tools | 11–14 | tools/clip_adapter_tool.py, sbi_tool.py, freqnet_tool.py, tool_registry.py |
| 4 | Ensemble, Early Stopping & Memory | 15–17 | utils/ensemble.py, core/early_stopping.py, core/memory.py |
| 5 | LLM Integration & Agent Brain | 18–22 | utils/ollama_client.py, core/prompts/forensic_summary.py, core/agent.py |
| 6 | Entry Points & User Interfaces | 23–25 | main.py, app.py (Streamlit), gradio_app.py |
| 7 | Testing & Integration | 26–30 | conftest.py, unit tests for tools, ensemble, agent, e2e integration test |

---

## PHASE 0: Project Scaffolding & Configuration (Days 1–2)

### Day 1: Core Configuration, Environment, and Logging

#### Prompt for Day 1:

**Section A: Context Reminder**
You are building Aegis-X, a 100% offline agentic deepfake detection system where a local LLM (Phi-3 Mini via Ollama) orchestrates 8 specialized forensic tools to produce an explainable verdict. 
This is Phase 0, Day 1: Project Scaffolding & Configuration. Nothing exists yet. We are setting up the foundational configuration, environment variables, dependencies, exception handling, and structured logging.

**Section B: Today's Objectives**
- Create `requirements.txt` to define pinned dependencies.
- Create `.env.example` as a template for environment variables.
- Create `core/config.py` containing all configuration dataclasses representing the single source of truth for the app.
- Create `core/exceptions.py` defining custom exception classes.
- Create `utils/logger.py` to set up standard logging.

**Section C: Detailed Specifications**
1. `requirements.txt`: 
   Define the following packages (use reasonable recent versions): `dlib`, `torch`, `torchvision`, `torchaudio`, `insightface`, `c2pa-python`, `scipy`, `numpy`, `opencv-python`, `streamlit`, `gradio`, `httpx`, `pydantic`, `python-dotenv`.

2. `.env.example`:
   Keys to include: `AEGIS_MODEL_DIR=models/`, `AEGIS_DEVICE=auto`, `OLLAMA_ENDPOINT=http://localhost:11434`, `LOG_LEVEL=INFO`.

3. `core/config.py`:
   Use `dataclasses` to define the configuration hierarchy:
   - `ModelPaths`: fields: `phi3_model` (str), `clip_adapter_weights` (str), `sbi_weights` (str), `freqnet_weights` (str), `dlib_shape_predictor` (str).
   - `AgentConfig`: fields: `max_retries` (int, default=3), `llm_timeout` (int, default=120), `ollama_endpoint` (str, default="http://localhost:11434").
   - `EnsembleWeights`: EXACT default values: `clip_adapter=0.30`, `sbi=0.20`, `freqnet=0.20`, `rppg=0.15`, `dct=0.10`, `geometry=0.03`, `illumination=0.02`. (Note: C2PA is binary and not weighted).
   - `ThresholdConfig`: EXACT default values: `real_threshold` (float, default=0.15), `fake_threshold` (float, default=0.85), `early_stop_confidence` (float, default=0.85). Optional: add `sbi_skip_clip_threshold` (default=0.70).
   - `PreprocessingConfig`: fields: `face_crop_size` (int, default=224), `sbi_crop_size` (int, default=380), `native_patch_size` (int, default=224), `max_video_frames` (int, default=300), `min_video_frames` (int, default=90), `extract_fps` (int, default=30).
   - `AegisConfig`: A master dataclass that groups all instances of the above configs (`models`, `agent`, `weights`, `thresholds`, `preprocessing`). Apply `python-dotenv` here to optionally override with available env vars.

4. `core/exceptions.py`:
   Define: 
   - `AegisError(Exception)` (base class)
   - `ModelLoadError(AegisError)`
   - `PreprocessingError(AegisError)`
   - `ToolExecutionError(AegisError)`

5. `utils/logger.py`:
   Implement `setup_logger(name: str)` that returns a standard `logging.Logger`. It should log to both `stdout` and a `logs/aegis.log` file, using a clean format: `"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`.

**Section D: Implementation Rules for That Day**
- Use Python 3.10+.
- Provide full type hints, return types, and Google-style docstrings for every class and method.
- `AegisConfig` must be easily instantiable with sensible defaults out-of-the-box.
- Use `pathlib.Path` only; do not use raw os/string paths.
- ONLY use the `logging` module — no `print()` statements.

**Section E: Testing & Verification Steps**
Create a `test_day1.py` script:
```python
from core.config import AegisConfig
from utils.logger import setup_logger

def test_day1():
    cfg = AegisConfig()
    log = setup_logger('test')
    log.info('Day 1 setup complete.')
    assert cfg.weights.clip_adapter == 0.30
    assert cfg.thresholds.fake_threshold == 0.85
    print("✅ Day 1 configuration and logging tests passed!")

if __name__ == "__main__":
    test_day1()
```
Run `python test_day1.py`.
Expected output: A log line printed to the console and appended to `logs/aegis.log`, followed by success message.

**Section F: Files Produced**
- `requirements.txt`
- `.env.example`
- `core/config.py`
- `core/exceptions.py`
- `utils/logger.py`
(Depends on: nothing. Enables: Consistent configuration and logging for the rest of the project.)

#### Day 1 Summary:
- Files: requirements.txt, .env.example, core/config.py, core/exceptions.py, utils/logger.py
- Depends on: nothing
- Enables: Day 2 (Base tools) and all future development.

---

### Day 2: Base Tool Protocol & Data Structures

#### Prompt for Day 2:

**Section A: Context Reminder**
Aegis-X is an offline, agentic multi-modal forensic engine. 
This is Phase 0, Day 2. So far, we have built the configuration (`core/config.py`), logging, and exceptions. Today we define the unified interface and payload contract for all 8 forensic tools. This contract is sacred.

**Section B: Today's Objectives**
- Create `core/data_types.py` holding the unified `ToolResult` interface.
- Create `core/base_tool.py` containing the `BaseForensicTool` abstract class that all tools will extend.

**Section C: Detailed Specifications**
1. `ToolResult` dataclass (in `core/data_types.py`):
   Must EXACTLY contain:
   - `tool_name`: str
   - `success`: bool
   - `score`: float (0.0 to 1.0, where 1.0 = maximally fake)
   - `confidence`: float (0.0 to 1.0)
   - `details`: dict (Raw diagnostic details/metrics)
   - `error`: str | None
   - `execution_time`: float (in seconds)
   - `evidence_summary`: str (A strictly formatted natural-language summary for the LLM)

2. `BaseForensicTool` class (in `core/base_tool.py`):
   - Inherits from `abc.ABC`.
   - Properties: `@property def tool_name(self) -> str:`
   - Abstract Method: `def setup(self) -> None:` (For loading models, verifying paths, etc.)
   - Abstract Method: `def _run_inference(self, input_data: Any) -> ToolResult:` 
   - Concrete Method: `def execute(self, input_data: Any) -> ToolResult:` 
     **Algorithm for `execute`:**
     1. Start a timer (`time.time()`).
     2. Open a `try` block and call `self._run_inference(input_data)`.
     3. Ensure the returned `ToolResult` has its `execution_time` set accurately.
     4. Catch *any* `Exception` (`except Exception as e`).
     5. On exception, log the error using `utils.logger` and return a failure `ToolResult` with EXACTLY:
        `success=False`, `score=0.5` (neutral so it doesn't skew the ensemble), `confidence=0.0` (so its weight becomes 0), `error=str(e)`, and `evidence_summary=f"Tool {self.tool_name} failed: {str(e)}"`.

**Section D: Implementation Rules for That Day**
- Fully typed, Google-style docstrings, Python 3.10+.
- Import the logger from `utils.logger.py` and use it to log the caught exceptions in `execute`.
- **Critical Rule**: No tool should ever crash the application — the `execute` wrapper is a mandatory firewall. If a tool fails (OOM, bad input, corrupted model), the agent logic must just ignore it smoothly.

**Section E: Testing & Verification Steps**
Create `test_day2.py` in the root:
```python
from core.base_tool import BaseForensicTool
from core.data_types import ToolResult

class DummyCrashTool(BaseForensicTool):
    @property
    def tool_name(self) -> str: return "dummy_crash"
    def setup(self) -> None: pass
    def _run_inference(self, input_data):
        raise RuntimeError("Simulated OOM crash")

def test_day2():
    tool = DummyCrashTool()
    result = tool.execute("dummy_input")
    assert not result.success
    assert result.score == 0.5
    assert result.confidence == 0.0
    assert "Simulated OOM crash" in result.error
    print("✅ Day 2 base tool error contract tested successfully! No crash occurred.")

if __name__ == "__main__":
    test_day2()
```
Run `python test_day2.py`.
Expected output: Validates the fallback `ToolResult` values and prints the success message. The script MUST NOT crash.

**Section F: Files Produced**
- `core/data_types.py`
- `core/base_tool.py`
- Depends on: `utils/logger.py`
- Enables: CPU and GPU tool implementations in Phases 2 and 3.

#### Day 2 Summary:
- Files: core/data_types.py, core/base_tool.py
- Depends on: utils/logger.py
- Enables: All tool construction in Phases 2 and 3.

---

## PHASE 1: Utilities & Preprocessing Pipeline (Days 3–5)

### Day 3: Image & Video Utilities

#### Prompt for Day 3:

**Section A: Context Reminder**
Aegis-X parses video and image inputs for its forensic pipeline. 
This is Phase 1, Day 3. You have `core/config.py` and base data structures. Today we build foundational I/O utilities for media.

**Section B: Today's Objectives**
- Create `utils/image.py` for image loading and simple validation.
- Create `utils/video.py` for safe video frame extraction.

**Section C: Detailed Specifications**
1. `utils/image.py` functions:
   - `load_image(path: Path) -> np.ndarray`: 
     Read image via `cv2.imread()`. If it fails or file does not exist, raise `FileNotFoundError`. Convert from BGR to RGB using `cv2.cvtColor`. Return as a numpy array.
   - `is_image(path: Path) -> bool`: Checks extension against a known list (jpg, png, webp, etc).

2. `utils/video.py` functions:
   - `extract_frames(path: Path, max_frames: int, extract_fps: int) -> list[np.ndarray]`:
     1. Use `cv2.VideoCapture` to open the video.
     2. Get native video FPS (`cap.get(cv2.CAP_PROP_FPS)`).
     3. Calculate frame skip interval to achieve the target `extract_fps` (e.g., Target=30fps, Native=60fps -> Skip every 2nd frame).
     4. Loop and read frames up to `max_frames`.
     5. Convert each BGR frame to RGB.
     6. Return the list of frame numpy arrays.
   - `get_video_duration(path: Path) -> float`: Return duration in seconds by dividing `CAP_PROP_FRAME_COUNT` by `CAP_PROP_FPS`.

**Section D: Implementation Rules for That Day**
- Enforce strict typing.
- Use `cv2` and `numpy`.
- Add docstrings explaining parameter roles (like `extract_fps`).
- Handle cases where paths do not exist cleanly.

**Section E: Testing & Verification Steps**
Create `test_day3.py`:
```python
from pathlib import Path
from utils.video import extract_frames

def test_day3():
    # Replace with a real path if available
    video_path = Path("sample.mp4")
    if video_path.exists():
        frames = extract_frames(video_path, max_frames=5, extract_fps=30)
        assert isinstance(frames, list)
        assert len(frames) <= 5
        assert frames[0].shape[-1] == 3 # RGB format
        print("✅ Day 3 media I/O passed!")
    else:
        print("⚠️ No sample.mp4 found, skipping array shape test.")

if __name__ == "__main__":
    test_day3()
```
Run `python test_day3.py`.
Expected output: Confirms frame limits and 3-channel (RGB) shape on a sample video.

**Section F: Files Produced**
- `utils/image.py`
- `utils/video.py`
- Depends on: OpenCV and Numpy.
- Enables: Advanced facial preprocessing.

#### Day 3 Summary:
- Files: utils/image.py, utils/video.py
- Depends on: None
- Enables: Day 4 Preprocessor

---

### Day 4: Preprocessing Pipeline

#### Prompt for Day 4:

**Section A: Context Reminder**
Aegis-X requires strictly formatted face crops and native resolution patches to feed its physics and GPU tools. Hand-crafting the spatial boundaries is critical to system success.
This is Phase 1, Day 4. We are leveraging `dlib` to build a 68-point landmark extractor and patch generator.

**Section B: Today's Objectives**
- Create `utils/preprocessing.py`.
- Define the `PreprocessResult` dataclass.
- Build the `Preprocessor` class using `dlib` to extract standard face crops and exact anatomical patches.

**Section C: Detailed Specifications**
1. `PreprocessResult` dataclass:
   Fields exactly as follows:
   - `has_face`: bool
   - `landmarks`: np.ndarray of shape `[68, 2]`, or None
   - `face_crop_224`: np.ndarray (for CLIP/FreqNet), or None
   - `face_crop_380`: np.ndarray (for SBI), or None
   - `patch_left_eye`: np.ndarray (224x224 crop centered on `landmarks[36:42]`), or None
   - `patch_right_eye`: np.ndarray (224x224 crop centered on `landmarks[42:48]`), or None
   - `patch_hairline`: np.ndarray (224x224 crop vertically above `landmarks[17:27]`), or None
   - `patch_jaw`: np.ndarray (224x224 crop covering `landmarks[0:17]`), or None
   - `frames_30fps`: list[np.ndarray], or None (All extracted frames if video)
   - `original_media_type`: str ("image" or "video")

2. `Preprocessor` class:
   - `__init__(self, config: PreprocessingConfig, shape_predictor_path: Path)`: 
     Load `dlib.get_frontal_face_detector()` and `dlib.shape_predictor(str(shape_predictor_path))`.
   - `_get_landmarks(self, image: np.ndarray) -> np.ndarray | None`: 
     Run detector. If face found, run predictor. Return exactly 68 `(x, y)` coordinates as a numpy array.
   - `_crop_align(self, image: np.ndarray, landmarks: np.ndarray, size: int) -> np.ndarray`: 
     Extract bounding box containing all 68 landmarks, add 20% margin, crop, and resize to `(size, size)`. Use `cv2.INTER_AREA` for downscaling.
   - `_extract_native_patches(self, image: np.ndarray, landmarks: np.ndarray) -> tuple`:
     Extract the 4 structural patches at original resolution, resize to 224x224. 
     - left_eye: tightly bound `landmarks[36:42]` with 30% margin.
     - right_eye: tightly bound `landmarks[42:48]` with 30% margin.
     - hairline: Region starting at the top of the forehead bounding box (`landmarks[17:27]`) extending upwards into the hair.
     - jaw: tightly bound `landmarks[0:17]`.
   - `process_media(self, path: Path) -> PreprocessResult`:
     Check if image or video. Use `utils.video` or `utils.image`. If video, run detection on the very first successful frame to get `landmarks`, and keep all frames for `frames_30fps`. Populate and return the `PreprocessResult`.

**Section D: Implementation Rules for That Day**
- Handle `ValueError` or missing face smoothly by returning `PreprocessResult` initialized to `has_face=False`.
- Utilize crop size constants from `PreprocessingConfig`.
- All outputs are RGB `np.ndarray` of dtype `uint8`.

**Section E: Testing & Verification Steps**
Create `test_day4.py`:
```python
from pathlib import Path
from core.config import PreprocessingConfig
from utils.preprocessing import Preprocessor

def test_day4():
    config = PreprocessingConfig()
    model_path = Path("models/shape_predictor_68_face_landmarks.dat")
    try:
        prep = Preprocessor(config, model_path)
        print("✅ Preprocessor initialized successfully.")
    except Exception as e:
        print(f"⚠️ Initialization failed (models missing?): {e}")

if __name__ == "__main__":
    test_day4()
```
Run `python test_day4.py`.
Expected output: Verifies smooth initializtion if dlib predictor is present, checks shapes of returned arrays downstream.

**Section F: Files Produced**
- `utils/preprocessing.py`
- Depends on: `core/config.py`, `utils/image.py`, `utils/video.py`, `dlib`
- Enables: Extracted crops to be fed into physics and GPU models downstream.

#### Day 4 Summary:
- Files: utils/preprocessing.py
- Depends on: core/config.py, utils/image.py, utils/video.py
- Enables: All Forensic Tools expecting standard crops and landmark coordinates.

---

### Day 5: VRAM Lifecycle Manager

#### Prompt for Day 5:

**Section A: Context Reminder**
Aegis-X operates in constrained consumer environments limited to a max peak VRAM of ~600MB. Only ONE GPU tool can be loaded at a time. The transition between tools requires absolute, deterministic memory clearing.
This is Phase 1, Day 5. Setting up robust sequential PyTorch VRAM lifecycle management.

**Section B: Today's Objectives**
- Create `utils/vram_manager.py` with device auto-detection logic.
- Create a deterministic `VRAMLifecycleManager` context manager that loads, yields, and ruthlessly purges models from GPU memory.

**Section C: Detailed Specifications**
1. `get_device() -> torch.device`: 
   Auto-detect sequentially:
   - If `torch.cuda.is_available()`, return `torch.device("cuda")`
   - If `torch.backends.mps.is_available()`, return `torch.device("mps")`
   - Else return `torch.device("cpu")`

2. `VRAMLifecycleManager` Context Manager:
   - `__init__(self, model_loader_function, *args, **kwargs)`: Store arguments.
   - `__enter__(self) -> torch.nn.Module`:
     1. Optional: Wait/verify no other GPU models are active (thread lock).
     2. Execute `model = self.model_loader_function(*self.args, **self.kwargs)`.
     3. Move model to `get_device()` using `model.to(device)`.
     4. Set `model.eval()`.
     5. Return model.
   - `__exit__(self, exc_type, exc_val, exc_tb)`:
     EXACT INSTRUCTIONS MUST BE FOLLOWED IN THIS ORDER:
     1. Unbind the `model` object from memory (`del self.model` if stored on instance).
     2. If device is CUDA: call `torch.cuda.empty_cache()`
     3. If device is MPS: call `torch.mps.empty_cache()`
     4. Force python garbage collection: `import gc; gc.collect()`

**Section D: Implementation Rules for That Day**
- Enforce strict adherence to resource clearing. PyTorch does not immediately release GPU RAM when variables go out of scope. The `del` + `empty_cache` + `gc.collect` combo is mandatory.
- The manager should act as a standard Python context manager.

**Section E: Testing & Verification Steps**
Create `test_day5.py`:
```python
import torch
from utils.vram_manager import VRAMLifecycleManager, get_device

def load_dummy_model():
    return torch.nn.Linear(5000, 5000)

def test_day5():
    device = get_device()
    print(f"Detected device: {device}")
    
    with VRAMLifecycleManager(load_dummy_model) as model:
        if device.type == "cuda":
            print(f"VRAM inside context: {torch.cuda.memory_allocated()} bytes")
            assert torch.cuda.memory_allocated() > 0
            
    if device.type == "cuda":
        print(f"VRAM outside context: {torch.cuda.memory_allocated()} bytes")
        assert torch.cuda.memory_allocated() == 0
        
    print("✅ Day 5 VRAM manager tested successfully! No memory leaks.")

if __name__ == "__main__":
    test_day5()
```
Run `python test_day5.py`.
Expected output: Memory usage should spike inside the context manager, and return precisely to 0 after exiting.

**Section F: Files Produced**
- `utils/vram_manager.py`
- Depends on: `torch`
- Enables: Safe sequential execution of Day 11-13 GPU tools without Out-of-Memory crashes.

#### Day 5 Summary:
- Files: utils/vram_manager.py
- Depends on: torch
- Enables: GPU Forensic Tools phase, enabling models to cycle through restricted VRAM constraint.

---
END OF PART 1. Say 'continue' for Part 2: Phase 2 (Days 6–10).
