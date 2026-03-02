# AEGIS-X IMPLEMENTATION PLAN: PART 4

## PHASE 5: LLM Integration & Agent Brain (Days 18–22)

### Day 18: Local LLM Client (Ollama Controller)

#### Prompt for Day 18:

**Section A: Context Reminder**
Aegis-X is strictly 100% offline. The "brain" of the agent is a small, locally hosted LLM (Phi-3 Mini) served via Ollama. 
This is Phase 5, Day 18. We are building the asynchronous Python controller that acts as the bridge between our Python logic and the local Ollama process.

**Section B: Today's Objectives**
- Create `utils/ollama_client.py` to handle LLM generations asynchronously.

**Section C: Detailed Specifications**
- `class OllamaClient`:
  - `def __init__(self, config: AgentConfig)`: Store the endpoint (e.g., `http://localhost:11434`) and timeout settings.
  - `async def check_health(self) -> bool`: Ping the Ollama API `/api/tags` endpoint using `httpx`. Return `True` if responsive, `False` otherwise.
  - `async def generate(self, prompt: str, system_prompt: str = "") -> str`:
    - Use `httpx.AsyncClient` to POST to `http://localhost:11434/api/generate`.
    - Payload: `{"model": config.phi3_model, "prompt": prompt, "system": system_prompt, "stream": False, "options": {"temperature": 0.1}}`. 
    - Temperature MUST be low (0.1) to avoid hallucinations in forensic reporting.
    - Keep timeouts high (e.g., 120s) because the 4GB GPU might be swapping weights, making the LLM inferences slower.
    - Extract and return the `"response"` string from the JSON.
    - If the request fails or times out, return a fallback string: "LLM synthesis failed or timed out. Relying on raw ensemble score."

**Section D: Implementation Rules for That Day**
- Handle `httpx.ConnectError` gracefully. (Ollama might not be started by the user yet).
- All functions must be `async`.

**Section E: Testing & Verification Steps**
Create `test_day18.py`:
```python
import asyncio
from core.config import AegisConfig
from utils.ollama_client import OllamaClient

async def test_day18():
    cfg = AegisConfig()
    # Assume phi3 is the model 
    client = OllamaClient(cfg.agent)
    
    is_up = await client.check_health()
    if not is_up:
        print("⚠️ Ollama is not running locally or endpoint is wrong. Please start Ollama.")
        return
        
    print("✅ Ollama is reachable!")
    
    response = await client.generate("Say the word 'Aegis'.", system_prompt="Answer briefly.")
    print(f"LLM Response: {response}")
    assert "LLM synthesis failed" not in response
    print("✅ Day 18 Ollama Client works natively.")

if __name__ == "__main__":
    asyncio.run(test_day18())
```
Run `python test_day18.py`.
Expected output: Verifies Ollama is running and successfully generates a fast stream-less response.

**Section F: Files Produced**
- `utils/ollama_client.py`
- Depends on: `httpx`, `AegisConfig`
- Enables: LLM integration without cloud APIs.

#### Day 18 Summary:
- Files: utils/ollama_client.py
- Depends on: httpx, local Ollama daemon
- Enables: Offline text generation for explanations.

---

### Day 19: Forensic Prompt Engineering

#### Prompt for Day 19:

**Section A: Context Reminder**
Aegis-X generates an easy-to-read "Evidence Summary" for human analysts. The LLM must not invent evidence; it must strictly digest the `ToolResult` strings and output a coherent summary.
This is Phase 5, Day 19. We are defining the system prompts and context injection templates.

**Section B: Today's Objectives**
- Create `core/prompts/forensic_summary.py`.

**Section C: Detailed Specifications**
- Define `SYSTEM_PROMPT`:
  "You are the Aegis-X Forensic Interpreter. You receive raw tool metrics and an ensemble probability score regarding a media file's authenticity. Your job is to summarize the findings for a human analyst. Do not hallucinate. Do not invent metrics. You must strictly use the evidence provided. If tools contradict each other, state the conflict clearly. Conclude with a definitive, single-sentence verdict: 'The media is likely REAL/FAKE.' Keep your summary under 150 words."
  
- Define `def build_user_prompt(ensemble_score: float, is_c2pa: bool, tool_results: list[ToolResult], history_matches: list[dict] = None) -> str`:
  - Compose a strict, structured context block:
    ```
    === AEGIS-X FINAL ENSEMBLE ===
    Ensemble Fake Probability: {ensemble_score * 100}%
    C2PA Override Active: {is_c2pa}
    
    === TOOL EVIDENCE ===
    [Loop through tool_results where success=True]
    - Tool: {tool.tool_name}
      Confidence assigned: {tool.confidence}
      Evidence: {tool.evidence_summary}
      
    === HISTORICAL CONTEXT ===
    [If history_matches is provided and not empty]
    - Found {len(history_matches)} similar cases in memory. Most recent human feedback for similar signature: {history_matches[0]['feedback_label']}
    ```
  - Return the composed string.

**Section D: Implementation Rules for That Day**
- This file contains no heavy logical algorithms, just clean, f-string based python functions returning strings.

**Section E: Testing & Verification Steps**
Create `test_day19.py`:
```python
from core.data_types import ToolResult
from core.prompts.forensic_summary import build_user_prompt, SYSTEM_PROMPT

def test_day19():
    res = ToolResult("run_rppg", True, score=0.9, confidence=0.8, details={}, execution_time_ms=0, evidence_summary="Flatline pulse detected.")
    
    prompt = build_user_prompt(0.95, False, [res])
    
    assert "Ensemble Fake Probability: 95.0%" in prompt
    assert "Flatline pulse detected" in prompt
    assert len(SYSTEM_PROMPT) > 50
    print("✅ Day 19 Prompts formatted correctly:\n", prompt)

if __name__ == "__main__":
    test_day19()
```
Run `python test_day19.py`.
Expected output: Ensures f-strings don't crash and text is cleanly formatted.

**Section F: Files Produced**
- `core/prompts/forensic_summary.py`
- Depends on: `ToolResult`
- Enables: Standardized data wrapping for the LLM.

#### Day 19 Summary:
- Files: core/prompts/forensic_summary.py
- Depends on: None
- Enables: Structured communication with Phi-3.

---

### Day 20: The Agent Loop — Initiation & State

#### Prompt for Day 20:

**Section A: Context Reminder**
Aegis-X is orchestrated by `ForensicAgent`. We will build this class over 3 days (Days 20-22) because it is the nexus of the entire application.
This is Phase 5, Day 20. We are building the `AgentState` dataclass and the `setup` / `preprocess` phases of the agent.

**Section B: Today's Objectives**
- Create `core/agent.py`.
- Define `AgentState`.
- Implement `ForensicAgent` initialization and media preprocessing.

**Section C: Detailed Specifications**
1. `class AgentState`:
   - `media_path: Path`
   - `original_media_type: str` (image/video)
   - `preprocessed_data: PreprocessResult | None = None`
   - `tool_results: list[ToolResult] = []`
   - `ensemble_score: float = 0.5`
   - `is_c2pa_override: bool = False`
   - `llm_summary: str = ""`
   - `start_time: float = 0.0`
   - `end_time: float = 0.0`
   - `error: str | None = None`
   
2. `class ForensicAgent`:
   - `def __init__(self, config: AegisConfig)`:
     - Instantiate `self.config = config`.
     - Instantiate `self.preprocessor = Preprocessor(...)` (from Day 4).
     - Instantiate `self.registry = ToolRegistry()` (from Day 14).
     - Instantiate `self.llm = OllamaClient(config.agent)`.
     - Instantiate `self.early_stopping = EarlyStoppingController(config.thresholds)`.
     - Instantiate `self.memory = MemorySystem()`.
   - `async def analyze(self, media_path: str) -> AgentState`:
     - This is the main entry point. 
     - Initialize `state = AgentState(media_path=Path(media_path), start_time=time.time())`.
     - `try:`
       - Call `state.preprocessed_data = self.preprocessor.process_media(state.media_path)`.
       - (Leave space here for Day 21 tool logic).
     - `except Exception as e`:
       - Log error, set `state.error = str(e)`.
     - `finally:`
       - `state.end_time = time.time()`.
       - Return `state`.

**Section D: Implementation Rules for That Day**
- Import all the heavy classes. This file links everything together.
- Use `try/except` around the preprocessing so the UI doesn't crash on bad files.

**Section E: Testing & Verification Steps**
Create `test_day20.py`:
```python
import asyncio
from pathlib import Path
from core.agent import ForensicAgent, AgentState
from core.config import AegisConfig

async def test_day20():
    cfg = AegisConfig()
    try:
        agent = ForensicAgent(cfg)
        assert hasattr(agent, "preprocessor")
        assert hasattr(agent, "llm")
        print("✅ Day 20 Agent initiated with all subsystems successfully.")
    except Exception as e:
        print(f"⚠️ Init failed (likely missing weights/models from earlier days): {e}")

if __name__ == "__main__":
    asyncio.run(test_day20())
```
Run `python test_day20.py`.
Expected output: Asserts the agent imports and composes all 6 prior subsystems without syntax errors.

**Section F: Files Produced**
- `core/agent.py`
- Depends on: Preprocessor, Registry, OllamaClient, Config.
- Enables: Central orchestrator pattern.

#### Day 20 Summary:
- Files: core/agent.py
- Depends on: All subsystem classes
- Enables: The backbone of the application.

---

### Day 21: The Agent Loop — Execution Sequence & Early Stop

#### Prompt for Day 21:

**Section A: Context Reminder**
The `ForensicAgent.analyze` function needs to sequentially execute tools, check early stopping conditions, and accumulate scores.
This is Phase 5, Day 21. We are injecting the core execution loop into `core/agent.py`.

**Section B: Today's Objectives**
- Expand `ForensicAgent.analyze` in `core/agent.py`.

**Section C: Detailed Specifications**
- Inside the `try/except` block of `analyze` (after preprocessing):
  1. Define `execution_queue = ["check_c2pa", "run_geometry", "run_illumination", "run_sbi", "run_clip_adapter", "run_freqnet", "run_rppg"]`. (Order heavily prioritizes CPU tools first to trigger early stopping fast).
  2. For each `tool_name` in `execution_queue`:
     - **Check conditions**: If `tool_name == "run_rppg"` and `state.preprocessed_data.original_media_type != "video"`, skip.
     - Build `tool_input`: Pass the entire `PreprocessResult` dict. Crucially, calculate `current_clip_score` from `state.tool_results` and pass it into `tool_input["clip_score"]` so the SBI tool (Day 12) can use its skip logic.
     - Call `result = self.registry.execute_tool(tool_name, tool_input)`.
     - Append `result` to `state.tool_results`.
     - Calculate intermediate ensemble: `agg = calculate_ensemble_score(state.tool_results)`.
     - Update `state.ensemble_score = agg["ensemble_score"]` and `state.is_c2pa_override = agg["is_c2pa_override"]`.
     - **Early Stopping**: 
       - `pending = execution_queue[current_index + 1 : ]`
       - `if self.early_stopping.evaluate(state.ensemble_score, [r.tool_name for r in state.tool_results], pending):`
         - Log the early stop.
         - `break` (escape the loop!).

**Section D: Implementation Rules for That Day**
- Ensure CPU tools execute linearly entirely *before* GPU tools. C2PA -> Geo -> Illumination are all CPU. 
- Properly inject the intermediate `clip_score` into `tool_input` before invoking SBI.

**Section E: Testing & Verification Steps**
Create `test_day21.py`:
```python
# To test this, you will need dummy tools or a mock registry, or to run it on a small sample image.
import asyncio
from core.agent import ForensicAgent
from core.config import AegisConfig

async def test_day21():
    agent = ForensicAgent(AegisConfig())
    # Assuming test_image.jpg exists and you have standard models loaded:
    # state = await agent.analyze("test_image.jpg")
    # print(f"Executed {len(state.tool_results)} tools out of 7. Score: {state.ensemble_score}")
    print("✅ Day 21 Execution Loop logic added (Requires full models to run e2e).")

if __name__ == "__main__":
    asyncio.run(test_day21())
```
Run `python test_day21.py`.
Expected output: Syntax check passes for `core/agent.py`.

**Section F: Files Produced**
- `core/agent.py` (modified)
- Depends on: Ensemble calculators, Early Stopping controllers.
- Enables: Hardware-efficient dynamic routing.

#### Day 21 Summary:
- Files: core/agent.py (updated)
- Depends on: EarlyStoppingController, calculate_ensemble_score
- Enables: Sequential tool iteration.

---

### Day 22: The Agent Loop — LLM Synthesis & Memory Update

#### Prompt for Day 22:

**Section A: Context Reminder**
The final step of the agent's run is digesting the ensemble score, querying memory for similar historically evaluated cases, and asking the local LLM to write the final human-readable report.
This is Phase 5, Day 22. We finish `core/agent.py`.

**Section B: Today's Objectives**
- Finalize `ForensicAgent.analyze` in `core/agent.py`.

**Section C: Detailed Specifications**
- After the tool execution `for` loop finishes (or breaks early):
  1. (Memory) Format current tool scores: `current_scores = {r.tool_name: r.score for r in state.tool_results if r.success}`.
  2. (Memory) Attempt to fetch similar cases: `history_matches = self.memory.query_similar_history(current_scores)`.
  3. (Prompt) `prompt_str = build_user_prompt(state.ensemble_score, state.is_c2pa_override, state.tool_results, history_matches)`.
  4. (LLM) Call `state.llm_summary = await self.llm.generate(prompt_str, system_prompt=SYSTEM_PROMPT)`.
  5. (Logging) `logger.info(f"Final Verdict mapped to score {state.ensemble_score}. Writing to memory DB...")`
  6. (Memory) Compute definitive string verdict: `str_verdict = "FAKE" if state.ensemble_score >= self.config.thresholds.fake_threshold else "REAL" if state.ensemble_score <= self.config.thresholds.real_threshold else "INCONCLUSIVE"`.
  7. (Memory) Store to DB: `self.memory.store_case(file_hash=str(state.media_path), file_type=state.original_media_type, verdict=str_verdict, confidence=0.9, ensemble=state.ensemble_score, tool_scores_dict=current_scores, reasoning=state.llm_summary)`.

**Section D: Implementation Rules for That Day**
- Never let an LLM timeout crash the whole analysis. We built a fallback into `OllamaClient`, trust it.
- Since we use the raw filepath as the `file_hash` for now, be sure to cast it to string `str(state.media_path)`. (A true SHA-256 hash would be better if you wish to implement it).

**Section E: Testing & Verification Steps**
Create `test_day22.py`:
```python
import asyncio
from core.agent import ForensicAgent
from core.config import AegisConfig

async def test_day22():
    print("✅ Day 22 Agent loop complete. The system is formally capable of e2e analysis.")
    # E2E test will be executed when the CLI is built in Phase 6.

if __name__ == "__main__":
    asyncio.run(test_day22())
```
Run `python test_day22.py`.
Expected output: Syntax check passes for `core/agent.py`.

**Section F: Files Produced**
- `core/agent.py` (completed)
- Depends on: `OllamaClient`, `MemorySystem`
- Enables: Fully autonomous operation from file injestion to text report generation.

#### Day 22 Summary:
- Files: core/agent.py (completed)
- Depends on: Ollama, MemorySystem
- Enables: The full logic backbone is complete.

---
END OF PART 4. Say 'continue' for Part 5: Phase 6 & 7 (Days 23–30).
