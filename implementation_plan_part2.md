# AEGIS-X IMPLEMENTATION PLAN: PART 2

## PHASE 2: CPU Forensic Tools (Days 6–10)

### Day 6: C2PA Provenance Tool

#### Prompt for Day 6:

**Section A: Context Reminder**
You are building Aegis-X, an agentic deepfake detection system. We have completed the base infrastructure, configuration, preprocessing (dlib crops), and `BaseForensicTool`. 
This is Phase 2, Day 6. We are building the first of the 5 CPU-only tools: The C2PA Provenance Tool. It's unique because it returns a binary verification rather than a floating-point anomaly score.

**Section B: Today's Objectives**
- Create `core/tools/c2pa_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `C2PATool(BaseForensicTool)`:
  - `@property tool_name`: return `"check_c2pa"`
  - `def setup(self)`: Import verification for `c2pa-python`.
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - `input_data` MUST contain `"media_path"` (a `Path` or `str` to the original image/video).
    - Use the `c2pa.read_file(media_path)` logic (from the `c2pa-python` library).
    - If valid signatures are found, extract the `signer` and `timestamp`.
    - Returns a `ToolResult`:
      - If signed: `success=True`, `score=0.0` (0% fake, it's verified real), `confidence=1.0`.
      - If NOT signed (or no C2PA data exists): `success=True`, `score=0.5` (neutral, absence of evidence is not evidence of absence), `confidence=0.0`.
      - `evidence_summary`: "Signed by [Signer] at [Timestamp]" OR "No C2PA provenance data found."
      - `details`: Dict containing raw extracted C2PA metadata if available.

**Section D: Implementation Rules for That Day**
- Handle the case where `c2pa.read_file` raises an exception (e.g., file not supported or missing library) — catch it natively within `_run_inference` and return the "neutral" `score=0.5` result rather than crashing. 

**Section E: Testing & Verification Steps**
Create `test_day6.py`:
```python
import tempfile
from pathlib import Path
from core.tools.c2pa_tool import C2PATool

def test_day6():
    tool = C2PATool()
    tool.setup()
    
    # Create an empty file to test the 'unsigned' logic
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        mock_path = Path(f.name)
        
    try:
        result = tool.execute({"media_path": mock_path})
        print(f"C2PA Result on unsigned file: {result}")
        assert result.success is True
        assert result.score == 0.5
        assert result.confidence == 0.0
        assert "No C2PA" in result.evidence_summary
        print("✅ Day 6 C2PA tool tested successfully! Neutral fallback works.")
    finally:
        mock_path.unlink()

if __name__ == "__main__":
    test_day6()
```
Run `python test_day6.py`.
Expected output: The tool gracefully returns a neutral score (`0.5`) with 0 confidence for an unsigned/empty image file.

**Section F: Files Produced**
- `core/tools/c2pa_tool.py`
- Depends on: `c2pa-python`, `BaseForensicTool`.
- Enables: The agent can rapidly clear content-credentialed images.

#### Day 6 Summary:
- Files: core/tools/c2pa_tool.py
- Depends on: c2pa-python
- Enables: Fast-path exiting for cryptographically signed media.

---

### Day 7: Biological Liveness Tool (rPPG)

#### Prompt for Day 7:

**Section A: Context Reminder**
Aegis-X uses physics and biological signals to defeat deepfakes. 
This is Phase 2, Day 7. Today we implement the POS (Plane Orthogonal to Skin-tone) rPPG algorithm. Generated faces lack a biological pulse. You are adapting the mathematical equations strictly from the Aegis-X README into `run_rppg`. **NEVER return a BPM.** Only return liveness confidence.

**Section B: Today's Objectives**
- Create `core/tools/rppg_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `RPPGTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_rppg"`
  - `def setup(self)`: No-op.
  - `def extract_pos_signal(self, frames: np.ndarray, fs: int) -> np.ndarray`:
    1. `frames` is shape `[N, H, W, 3]`. Take spatial mean per frame -> `RGB` array `[N, 3]`.
    2. Normalize over a sliding window (1.6 seconds). `Rn=R/mean(R)`.
    3. Project: `S1 = Gn - Bn` and `S2 = Gn + Bn - 2*Rn`.
    4. Compute `H = S1 + (std(S1)/std(S2)) * S2`.
    5. Filter `H` using `scipy.signal.butter` bandpass (0.75 – 4.0 Hz).
    6. Return `H` (the 1D BVP signal).
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - `input_data` MUST contain `"frames_30fps"` (list of crops, specifically the forehead ROI if provided by the preprocessor).
    - Requirements: `min_frames = 90`. If `<90`, return neutral result (`score=0.5, conf=0.0`).
    - Run `extract_pos_signal` on the frames.
    - Calculate Variance: `var = float(np.var(bvp))`.
    - Calculate SNR: power in the dominant frequency peak vs noise floor using `scipy.signal.periodogram`.
    - Thresholds: If SNR > 3.0 AND variance > 0.020 -> Liveness present.
    - Result logic:
      - If Liveness Present: `score=0.0` (Real), `confidence = min(1.0, SNR / 10.0)`.
      - If Flatline (variance < 0.005): `score=1.0` (Fake), `confidence = 0.8`.
      - Else (Ambiguous): `score=0.5`, `confidence=0.1`.
    - `evidence_summary`: Explain if physiological variation was found or if it was a flatline.

**Section D: Implementation Rules for That Day**
- Only use `scipy` and `numpy`.
- DO NOT return or log the BPM. The UI explicitly forbids showing corrupted BPMs; it only cares about the *presence* of the biological signal.

**Section E: Testing & Verification Steps**
Create `test_day7.py`:
```python
import numpy as np
from core.tools.rppg_tool import RPPGTool

def test_day7():
    tool = RPPGTool()
    
    # Create fake "flatline" video data (100 frames of 50x50 RGB noise with 0 biological variation)
    np.random.seed(42)
    fake_frames = [np.random.randint(100, 110, (50, 50, 3), dtype=np.uint8) for _ in range(100)]
    
    result = tool.execute({"frames_30fps": fake_frames})
    
    print(f"rPPG Output: Score={result.score}, Conf={result.confidence}")
    print(f"Summary: {result.evidence_summary}")
    
    assert result.success is True
    assert result.score > 0.8  # Should flag as fake due to flatline
    print("✅ Day 7 rPPG tool correctly identified flatline synthetic data!")

if __name__ == "__main__":
    test_day7()
```
Run `python test_day7.py`.
Expected output: The POS algorithm processes the random noise, detects no periodic pulse (flatline), and assigns a high fake score.

**Section F: Files Produced**
- `core/tools/rppg_tool.py`
- Depends on: `scipy`, `numpy`
- Enables: Liveness detection for video inputs.

#### Day 7 Summary:
- Files: core/tools/rppg_tool.py
- Depends on: scipy, numpy
- Enables: Real-time pulse extraction from webcam/video footage.

---

### Day 8: DCT Frequency Analysis Tool

#### Prompt for Day 8:

**Section A: Context Reminder**
We are building Aegis-X. Deepfakes undergo multiple compression cycles on social media.
This is Phase 2, Day 8. We are building the DCT Frequency Analysis Tool. By analyzing the 8x8 block Discrete Cosine Transform (DCT) histogram, we can detect double-quantization signals characteristic of re-compressed deepfakes.

**Section B: Today's Objectives**
- Create `core/tools/dct_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `DCTTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_dct"`
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    1. Read `input_data["face_crop_224"]` (RGB numpy array).
    2. Convert to grayscale float64 using `cv2.cvtColor`.
    3. Ensure dimensions are divisible by 8 (pad or crop).
    4. Group into 8x8 blocks. Reshape the entire array: `blocks = image.reshape(h//8, 8, w//8, 8).transpose(0, 2, 1, 3)`.
    5. Apply 2D DCT on every block: `scipy.fftpack.dct` (or `scipy.fft.dctn` with `norm='ortho'`).
    6. Flatten the AC coefficients (ignore the [0,0] DC component of each 8x8 block).
    7. Compute the histogram of these coefficients. Find periodic peaks via autocorrelation:
       `autocorr = np.correlate(hist, hist, mode='same')`
    8. Measure double quantization: The amplitude of secondary peaks relative to the primary peak. If `peak_ratio > 0.15`, flag.
    9. Calculate `score`: `min(1.0, peak_ratio / 0.30)`.
    10. `confidence`: `min(0.9, score + 0.2)`
    11. `evidence_summary`: "DCT analysis detected double-quantization artifacts indicating structural modification" or "Smooth DCT frequency distribution consistent with natural imagery."

**Section D: Implementation Rules for That Day**
- Use `scipy.fftpack.dct` or `scipy.fft.dctn`.
- This is a mathematical transformation, handle zeros or `NaNs` safely (add `1e-10` to denominators).

**Section E: Testing & Verification Steps**
Create `test_day8.py`:
```python
import numpy as np
import cv2
from core.tools.dct_tool import DCTTool

def test_day8():
    tool = DCTTool()
    
    # Generate a pure white image (no texture, no double quant expected)
    clean_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
    
    # Generate an image with harsh 8x8 blocking artifacts
    blocky_img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            blocky_img[i:i+8, j:j+8] = 255
            
    res_clean = tool.execute({"face_crop_224": clean_image})
    res_blocky = tool.execute({"face_crop_224": blocky_img})
    
    print(f"Clean Score: {res_clean.score}, Blocky Score: {res_blocky.score}")
    assert res_clean.success and res_blocky.success
    # Blocky image should yield a higher artifact score than the perfectly smooth one
    assert res_blocky.score > res_clean.score
    print("✅ Day 8 DCT tool tested. Differentiates smooth vs high-frequency grid artifacts.")

if __name__ == "__main__":
    test_day8()
```
Run `python test_day8.py`.
Expected output: Success message confirming the algorithm differentiates smooth vs hard-gradient images.

**Section F: Files Produced**
- `core/tools/dct_tool.py`
- Depends on: `scipy.fft`.
- Enables: Analysis of compression-surviving artifacts.

#### Day 8 Summary:
- Files: core/tools/dct_tool.py
- Depends on: scipy.fft
- Enables: Detection of compression and quantization artifacts.

---

### Day 9: Geometric Physics Tool

#### Prompt for Day 9:

**Section A: Context Reminder**
Aegis-X checks whether facial structures adhere to anatomical physics. Generative models hallucinative visual textures but often fail basic 3D human constraints. 
This is Phase 2, Day 9. We are implementing 7 explicit anthropometric measurements on the extracted `dlib` landmarks. This tool is a cornerstone of our zero-shot physics defense.

**Section B: Today's Objectives**
- Create `core/tools/geometry_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `GeometryTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_geometry"`
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - Reads `landmarks` (np.ndarray of shape [68, 2]) from `input_data`. If None, return neutral result `score=0.5`.
    - Function: `def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))`
    - You MUST implement EXACTLY these 7 distinct checks using `dist`:
      1. **IPD ratio**: `dist(landmarks[36:42].mean(0), landmarks[42:48].mean(0)) / dist(landmarks[0], landmarks[16])`. Valid: 0.42 to 0.52.
      2. **Philtrum ratio**: `dist(landmarks[33], landmarks[51]) / dist(landmarks[27], landmarks[8])`. Valid: 0.10 to 0.15.
      3. **Eye width asymmetry**: `abs(dist(lm[36], lm[39]) - dist(lm[42], lm[45])) / width`. Flag if > 0.05.
      4. **Jaw yaw symmetry**: Anchor at nose (`lm[27]`). `abs(dist(lm[27], lm[0]) - dist(lm[27], lm[16])) / width`. Flag if > 0.08. Skip check if face pose is profiled (skip if `yaw_proxy > 0.15`).
      5. **Nose width ratio**: `dist(lm[31], lm[35]) / IPD`. Valid: 0.55 to 0.70.
      6. **Mouth width ratio**: `dist(lm[48], lm[54]) / IPD`. Valid: 0.85 to 1.05.
      7. **Vertical thirds**: `upper = lm[19] to lm[27]`, `mid = lm[27] to lm[33]`, `lower = lm[33] to lm[8]`. Check if any deviate by > 15% from the average third.
    - Each check that fails adds a string to a `violations` list.
    - `fake_score = len(violations) / 7.0`
    - `confidence = 0.8` (if landmarks exist).
    - `evidence_summary`: Include strings of EXACTLY which anatomical structures are violating constraints (e.g., "Violations found in Jaw yaw symmetry and Philtrum ratio.").

**Section D: Implementation Rules for That Day**
- Use precise indexing on the [68, 2] array.
- Catch `ZeroDivisionError` natively with `+ 1e-10` on all denominators.

**Section E: Testing & Verification Steps**
Create `test_day9.py`:
```python
import numpy as np
from core.tools.geometry_tool import GeometryTool

def test_day9():
    tool = GeometryTool()
    
    # Generate a PERFECTLY symmetrical dummy face grid matching the expected bounds roughly
    landmarks = np.zeros((68, 2))
    landmarks[0] = [0, 50]       # left jaw
    landmarks[16] = [100, 50]    # right jaw
    landmarks[36] = [25, 25]     # left eye outer
    landmarks[39] = [40, 25]     # left eye inner
    landmarks[42] = [60, 25]     # right eye inner
    landmarks[45] = [75, 25]     # right eye outer
    landmarks[27] = [50, 40]     # nose bridge
    landmarks[31] = [40, 60]     # left nostril
    landmarks[33] = [50, 60]     # nose tip
    landmarks[35] = [60, 60]     # right nostril
    landmarks[51] = [50, 65]     # lip top
    landmarks[48] = [30, 70]     # mouth left
    landmarks[54] = [70, 70]     # mouth right
    landmarks[8] = [50, 100]     # chin
    landmarks[19] = [30, 10]     # left brow
    
    result = tool.execute({"landmarks": landmarks})
    print(f"Geometry Score: {result.score}")
    assert result.success is True
    assert "violations" in result.details
    print("✅ Day 9 Geometry tool evaluated layout array successfully!")

if __name__ == "__main__":
    test_day9()
```
Run `python test_day9.py`.
Expected output: The algorithm parses the synthetic array, calculates ratios, determines what passed/failed, and outputs a score without IndexErrors.

**Section F: Files Produced**
- `core/tools/geometry_tool.py`
- Depends on: valid [68, 2] landmarks from Preprocessor.
- Enables: High success rate against AI generators that fail 3D anatomical layout.

#### Day 9 Summary:
- Files: core/tools/geometry_tool.py
- Depends on: Preprocessor (dlib landmarks)
- Enables: Zero-shot anatomical physics verification.

---

### Day 10: Illumination Physics Tool

#### Prompt for Day 10:

**Section A: Context Reminder**
Diffusion models generate photorealistic faces, but they often composite them into scenes with conflicting directional light sources. 
This is Phase 2, Day 10. We are using Horn's "Shape-from-Shading" logic to compare the dominant illumination gradient of the face against the immediate background.

**Section B: Today's Objectives**
- Create `core/tools/illumination_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `IlluminationTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_illumination"`
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    1. Read `input_data["face_crop_224"]` and `input_data["landmarks"]`. If either missing, return neutral score.
    2. Convert RGB crop to `YCrCb` and isolate the `Y` (Luma) channel.
    3. `midpoint_x = int(landmarks[27:34, 0].mean())` -> Calculate the vertical spine of the nose to split the face left and right.
    4. Calculate Face Lighting: Mean Luma of the Left side of the face vs Mean Luma of the Right side of the face.
       `face_grad = abs(face_l - face_r) / (face_l + face_r + 1e-6)`.
    5. Interpret diffuse lighting: If `face_grad < 0.05`, the lighting is diffuse (straight-on flat lighting). Return `score=0.2` (typically real).
    6. Calculate Scene Context (Simplified 2D bounds): Split the bottom 20 rows of the crop into left and right halves (proxy for neck/shoulders). `ctx_l` vs `ctx_r`.
    7. Which is brighter?
       `face_dom = "left" if face_l > face_r else "right"`
       `ctx_dom = "left" if ctx_l > ctx_r else "right"`
    8. Calculate mismatch penalty:
       - If `face_dom == ctx_dom`: `fake_score = face_grad * 0.20`
       - If mismatch (`!=`): `fake_score = 0.30 + (face_grad * 0.70)`
    9. `confidence`: `min(0.9, face_grad * 10)` (High gradient = strong lighting = confident result).
    10. `evidence_summary`: "Face illumination direction mismatches environmental scene context" or "Consistent lighting found."

**Section D: Implementation Rules for That Day**
- Use standard `numpy` slicing to divide the face. Wait to do this until converting the image to grayscale/luma.
- Always include `1e-6` on denominators.

**Section E: Testing & Verification Steps**
Create `test_day10.py`:
```python
import numpy as np
from core.tools.illumination_tool import IlluminationTool

def test_day10():
    tool = IlluminationTool()
    
    # Create a dummy image 224x224, 3 channel
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Bright left face
    dummy_img[50:150, 50:100] = 200
    # Dark right face
    dummy_img[50:150, 100:150] = 50
    
    # Context (neck area) MISMATCH: Bright right, dark left
    dummy_img[200:220, 50:100] = 50
    dummy_img[200:220, 100:150] = 200
    
    dummy_landmarks = np.zeros((68, 2))
    dummy_landmarks[27:34, 0] = 100  # Sets the midline X to 100
    
    res = tool.execute({
        "face_crop_224": dummy_img,
        "landmarks": dummy_landmarks
    })
    
    print(f"Illumination Mismatch Score: {res.score}")
    assert res.success is True
    # The intentional mismatch should trigger a fake_score > 0.30
    assert res.score > 0.30
    print("✅ Day 10 Illumination tool caught the synthetic face-scene mismatch!")

if __name__ == "__main__":
    test_day10()
```
Run `python test_day10.py`.
Expected output: The algorithm processes the starkly mismatched array and produces a high fake penalty score.

**Section F: Files Produced**
- `core/tools/illumination_tool.py`
- Depends on: cv2 (for YCrCb conversion)
- Enables: Last of the 5 CPU-bound, instant forensic tools.

#### Day 10 Summary:
- Files: core/tools/illumination_tool.py
- Depends on: cv2, numpy
- Enables: Mismatch detection against Diffusion models.

---
END OF PART 2. Say 'continue' for Part 3: Phase 3 (Days 11–14).
