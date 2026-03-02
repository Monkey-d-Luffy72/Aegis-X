# AEGIS-X IMPLEMENTATION PLAN: PART 5

## PHASE 6: Entry Points & User Interfaces (Days 23–25)

### Day 23: CLI Application

#### Prompt for Day 23:

**Section A: Context Reminder**
Aegis-X provides 3 ways to interact with the system: CLI, Streamlit (for Analysts), and Gradio (for public Demos). 
This is Phase 6, Day 23. We are building the `main.py` CLI script, which is the fastest, lowest-overhead way to run the agent.

**Section B: Today's Objectives**
- Create `main.py` at the project root.

**Section C: Detailed Specifications**
- `main.py`:
  - Use `argparse` to accept CLI arguments:
    - `media_path` (required): Path to the image/video to analyze.
    - `--config`: Path to an optional custom YAML config file.
    - `--debug`: Boolean flag to set `LOG_LEVEL=DEBUG`.
    - `--json_out`: Boolean flag. If true, prints only a raw JSON string to stdout (useful for piping to other scripts).
  - Main workflow:
    1. Parse args, set up logger based on debug flag.
    2. Print a nice ASCII art logo for Aegis-X.
    3. Load `AegisConfig()`.
    4. Instantiate `ForensicAgent(config)`.
    5. Call `state = asyncio.run(agent.analyze(args.media_path))`
    6. If `args.json_out` is False, use the `rich` library to print a formatted table showing: Tool Name, Score, Confidence, and Summary. Below the table, print the Final Ensemble Score, the C2PA Override status, and the LLM's natural language verdict.
    7. If `args.json_out` is True, print `json.dumps` of the state.

**Section D: Implementation Rules for That Day**
- Ensure graceful handling of `KeyboardInterrupt` to abort safely.
- Do not let `rich` or standard `print` statements pollute the output if `--json_out` is true.

**Section E: Testing & Verification Steps**
Create `test_day23.py` (Mocking the CLI via subprocess):
```python
import subprocess

def test_day23():
    print("Testing CLI help text...")
    result = subprocess.run(["python", "main.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "media_path" in result.stdout
    print("✅ Day 23 CLI accepted arguments perfectly.")

if __name__ == "__main__":
    test_day23()
```
Run `python test_day23.py`.
Expected output: Ensures `main.py` loads instantly without syntax errors and correctly handles arguments.

**Section F: Files Produced**
- `main.py`
- Depends on: `rich`, `ForensicAgent`
- Enables: Headless, pipable usage on remote servers.

#### Day 23 Summary:
- Files: main.py
- Depends on: rich, argparse
- Enables: Direct script ingestion and power-user interface.

---

### Day 24: Analytical Dashboard (Streamlit)

#### Prompt for Day 24:

**Section A: Context Reminder**
The primary interface for intelligence analysts using Aegis-X is a locally hosted Streamlit dashboard. It exposes sliders to alter mathematical thresholds live and visually displays crop/landmark extraction.
This is Phase 6, Day 24. We are building `app.py`.

**Section B: Today's Objectives**
- Create `app.py`.

**Section C: Detailed Specifications**
- `app.py`:
  - Import `streamlit as st`.
  - Set page config: `st.set_page_config(page_title="Aegis-X Forensic Dashboard", layout="wide")`
  - Sidebar:
    - Display System Status (Ollama connection, GPU VRAM available via `utils.device`).
    - Expose settings for `AegisConfig.thresholds` adjusting `fake_threshold` and `early_stop_confidence`.
  - Main Panel:
    - File Uploader for images/videos.
    - If file uploaded -> show preview and an "Analyze Media" button.
    - Run `agent.analyze(st.session_state.file_path)` inside a top-level `st.spinner("Executing Forensic Suite...")`.
  - Results viewing:
    - Top metric cards: Final Ensemble Probability, Execution Time, and LLM's Verdict string.
    - Create a 2-column layout:
      - Left Col: The raw image/video.
      - Right Col: Expandable accordions for each ToolResult executed. If the tool is a GPU tool, badge it as `[GPU]`. Show the score on a colored progress bar (Green=0, Red=100%).

**Section D: Implementation Rules for That Day**
- Streamlit's async support can be tricky. You must wrap the `agent.analyze()` call properly using `asyncio.run()`.
- Use `st.session_state` to prevent re-running the heavy models every time an analyst adjusts a UI slider on the left.

**Section E: Testing & Verification Steps**
Create `test_day24.py`:
```python
import subprocess

def test_day24():
    print("Verifying Streamlit app loads...")
    # Just checking for syntax errors, not actually connecting to the browser
    result = subprocess.run(["python", "-m", "py_compile", "app.py"], capture_output=True)
    assert result.returncode == 0
    print("✅ Day 24 Streamlit interface syntax verified.")

if __name__ == "__main__":
    test_day24()
```
Run `python test_day24.py` AND try running `streamlit run app.py` manually in your terminal.
Expected output: The browser opens the dashboard UI with sidebars correctly populated.

**Section F: Files Produced**
- `app.py`
- Depends on: `streamlit`
- Enables: Main analyst GUI.

#### Day 24 Summary:
- Files: app.py
- Depends on: streamlit
- Enables: Visual, threshold-tweakable UI.

---

### Day 25: Public Demo Interface (Gradio)

#### Prompt for Day 25:

**Section A: Context Reminder**
For public instances where we want a simpler layout without sidebar knobs, Gradio is preferred. 
This is Phase 6, Day 25. We are building `gradio_app.py`.

**Section B: Today's Objectives**
- Create `gradio_app.py`.

**Section C: Detailed Specifications**
- `gradio_app.py`:
  - Import `gradio as gr`.
  - Use `gr.Blocks(theme=gr.themes.Soft())`.
  - Two primary tabs: "Image Analysis" and "Video Analysis".
  - Layout:
    - Row 1: Upload box on the left, "Output Verdict" markdown box + JSON summary on the right.
    - Wrap the `agent.analyze` call in a wrapper function that takes the file path and returns strings/dataframes formatting the `AgentState`.
  - Use `gr.Dataframe` to render the `ToolResults` in a clean table format.

**Section D: Implementation Rules for That Day**
- Keep the interface dead simple. Upload -> Analyze -> Table + Text. Minimal states.

**Section E: Testing & Verification Steps**
Create `test_day25.py`:
```python
import subprocess

def test_day25():
    print("Verifying Gradio app syntax...")
    result = subprocess.run(["python", "-m", "py_compile", "gradio_app.py"], capture_output=True)
    assert result.returncode == 0
    print("✅ Day 25 Gradio app syntax verified.")

if __name__ == "__main__":
    test_day25()
```
Run `python test_day25.py`.
Expected output: The code compiles.

**Section F: Files Produced**
- `gradio_app.py`
- Depends on: `gradio`
- Enables: Fast, zero-config public sharing.

#### Day 25 Summary:
- Files: gradio_app.py
- Depends on: gradio
- Enables: Web API and minimal UI setup.

---

## PHASE 7: Testing, Polish & Deployment (Days 26–30)

### Day 26: Pytest Fixtures & Component Tests

#### Prompt for Day 26:

**Section A: Context Reminder**
Robust pipelines require automated testing. 
This is Phase 7, Day 26. We are writing the `pytest` scaffold and unit tests for utilities.

**Section B: Today's Objectives**
- Create `tests/conftest.py`.
- Create `tests/test_utils.py` and `tests/test_ensemble.py`.

**Section C: Detailed Specifications**
1. `tests/conftest.py`:
   - Create pytest fixtures that return a dummy image `np.ndarray` and a dummy video (create an mp4 locally using `cv2.VideoWriter`).
   
2. `tests/test_utils.py`:
   - Test `utils.video.extract_frames`
   - Test `utils.preprocessing.Preprocessor._get_landmarks`. Test behavior when NO face is present.

3. `tests/test_ensemble.py`:
   - Bring over the logic from `test_day15.py` into formal `assert` statements within `pytest`.
   - Test standard arrays, conflicting confidence ratios, and the absolute C2PA override function.

**Section D: Implementation Rules for That Day**
- Tests must pass without GPUs. Do not initialize Torch datasets or models in `conftest`.

**Section E: Testing & Verification Steps**
Run `pytest tests/test_ensemble.py -v`.
Expected output: All test cases pass.

**Section F: Files Produced**
- `tests/conftest.py`, `tests/test_utils.py`, `tests/test_ensemble.py`
- Depends on: `pytest`
- Enables: CI/CD integration.

#### Day 26 Summary:
- Files: Pytest scaffolding and math unit testing.
- Depends on: pytest

---

### Day 27: E2E Integration Test

#### Prompt for Day 27:

**Section A: Context Reminder**
To ensure regressions don't break the agent loop, we need a Mock orchestration test.
This is Phase 7, Day 27.

**Section B: Today's Objectives**
- Create `tests/test_agent_e2e.py`.

**Section C: Detailed Specifications**
- We don't want to load Gigabytes of weights during CI loops.
- Create a `MockRegistry` that mimics `ToolRegistry` but returns hardcoded scores instantly.
- Inject the `MockRegistry` into `ForensicAgent`.
- Run the entire `agent.analyze()` pipeline.
- Assert that Early Stopping triggers exactly when it should given the mock scores.

**Section D: Implementation Rules for That Day**
- Use standard `unittest.mock` to monkeypatch `OllamaClient` so the test doesn't fail if Ollama daemon is offline.

**Section E: Testing & Verification Steps**
Run `pytest tests/test_agent_e2e.py -v`.
Expected output: Asserts the agent properly runs, queries tools, skips via early stop, and outputs state.

**Section F: Files Produced**
- `tests/test_agent_e2e.py`

#### Day 27 Summary:
- Files: tests/test_agent_e2e.py

---

### Day 28: Dockerization

#### Prompt for Day 28:

**Section A: Context Reminder**
Many users deploy on cloud compute. Docker streamlines it.
This is Phase 7, Day 28.

**Section B: Today's Objectives**
- Create `Dockerfile` and `docker-compose.yml`.

**Section C: Detailed Specifications**
- `Dockerfile`:
  - Use `nvidia/cuda:12.1.0-runtime-ubuntu22.04` as the base image.
  - Install python3.10 and pip.
  - Use `--mount=type=cache` with pip to speed up reinstalls.
  - `CMD ["streamlit", "run", "app.py"]`
- `docker-compose.yml`:
  - Define `aegis-core`: the python build.
  - Define `aegis-ollama`: attach the official Ollama docker image `ollama/ollama`. Ensure `aegis-core` can reach it on port `11434`.

**Section D: Implementation Rules for That Day**
- Keep image layers lean.

**Section E: Testing & Verification Steps**
Run `docker compose build`.
Expected output: The dockerfile downloads dependencies and packages the code successfully.

**Section F: Files Produced**
- `Dockerfile`, `docker-compose.yml`

#### Day 28 Summary:
- Files: Dockerfile, docker-compose.yml

---

### Day 29: Model Download Script

#### Prompt for Day 29:

**Section A: Context Reminder**
Aegis-X requires pre-trained PyTorch weights (FreqNet, SBI, ClipAdapter, Dlib).
This is Phase 7, Day 29. We need to automate fetching them before first run.

**Section B: Today's Objectives**
- Create `scripts/download_models.py`.

**Section C: Detailed Specifications**
- Use `requests` to pull weights from specified URLs into the `models/` directory.
- Use `tqdm` to show progress bars.
- Use `pathlib` to check if file sizes match and skip downloads if they already exist.

**Section D: Implementation Rules for That Day**
- The script should run entirely without `torch` or other heavy loads installed so it can be called first.

**Section E: Testing & Verification Steps**
Run `python scripts/download_models.py`.
Expected output: Logs the initiation of downloads.

**Section F: Files Produced**
- `scripts/download_models.py`

#### Day 29 Summary:
- Files: scripts/download_models.py

---

### Day 30: Documenting and Final Polish

#### Prompt for Day 30:

**Section A: Context Reminder**
We are at the end of the project.
This is Phase 7, Day 30.

**Section B: Today's Objectives**
- Clean up docstrings in all files.
- Re-run all verification steps to ensure nothing broke across iterations.
- Run `flake8` or `black` on the repository optionally.
- Mark the project stable.

**Section E: Testing & Verification Steps**
Run `pytest` against the entire suite. Launch the Streamlit and CLI interfaces manually.
Expected output: A pristine, functional, offline deepfake detection engine.

---
**END OF IMPLEMENTATION PLAN**
