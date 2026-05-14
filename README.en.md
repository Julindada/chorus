# Chorus

A psychology-driven multi-agent decision support system built on LangGraph.

Rather than giving advice from the outside, Chorus simulates the inner conflict and integration process of human psychology during decision-making. Seven agents — each representing a distinct psychological dimension — evaluate the user's situation in parallel. When disagreement is strong, they enter a cyclic debate. The final output is a decision report audited for consensus.

## 7 Psychological Dimensions

| Agent | Psychological Basis | Core Function |
|-------|-------------------|---------------|
| Arbiter (逻辑法官) | System 2 / Prefrontal Cortex | Cost-benefit analysis, probability judgment |
| Empath (情绪侦探) | System 1 / Appraisal Theory | Uncover hidden emotional signals; identify fear and excitement |
| Soothsayer (躯体预言家) | Somatic Marker Hypothesis (Damasio) | Predict impact on stress, energy, and sleep |
| Compass (意义向导) | Logotherapy (Frankl) | Evaluate alignment with life narrative and long-term goals |
| Narrator (自我叙述者) | Narrative Identity Theory (McAdams) | Detect conflict with self-concept |
| Conscience (良知证人) | Schwartz Value Vector | Report deviation between the decision and personal values — without judging |
| Guardian (关系守护者) | Attachment Theory | Assess impact on family, partner, and core social relationships |

## Prerequisites

**Required API Keys**

| Variable | Purpose | Where to get |
|----------|---------|--------------|
| `LLM_API_KEY` | LLM (Bailian / Qwen) | console.aliyun.com/bailian |
| `TAVILY_API_KEY` | Reality-check web search | app.tavily.com |
| `LANGSMITH_API_KEY` | LangGraph Studio tracing (optional) | smith.langchain.com |

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

## How to Run

### Option 1: Local Installation

**Requirements:**
- Python 3.11+
- uv (`pip install uv` or `brew install uv`)

```bash
uv sync
uv run chorus
```

### Option 2: Docker

**Requirements:**
- Docker

```bash
docker compose run --rm chorus
```

User profiles and reports are persisted in the Docker volume `chorus_data` and survive container removal.

**Retrieve a report from the volume:**

```bash
docker compose run --rm chorus cat /home/appuser/data/reports/<filename>.md > report.md
```

## Usage

After starting, follow the interactive prompts:

1. **Enter username** — used to load and save your Schwartz value profile
2. **First run** — choose questionnaire answers (recommended), paste JSON, or enter scores manually
3. **Describe your decision** — the more detail, the more accurate the analysis
4. **Enter candidate options** — one per line, blank line to finish (at least 2)
5. **Wait for analysis** — 7 agents evaluate in parallel; debate rounds start automatically when disagreement is high
6. **View the report** — after analysis, you can save it as a Markdown file

### Building Your Value Profile

The easiest path is to fill out `doc/chorus_values_questionnaire.pdf` (21 questions, ~5 min), then choose one of:

- **Enter answers directly in Chorus** — select "问卷答案（推荐）" at startup and type the 21 answers (1–6). You can paste all 21 space-separated on one line.
- **Ask an AI to compute it** — the PDF's last page includes a ready-to-copy AI prompt template.
- **Paste JSON directly** — select "粘贴 JSON" and paste the computed result.

**Schwartz value vector JSON format:**

```json
{
  "self_direction": 0.8,
  "stimulation": 0.6,
  "hedonism": 0.4,
  "achievement": 0.9,
  "power": 0.3,
  "security": 0.7,
  "conformity": 0.2,
  "tradition": 0.3,
  "benevolence": 0.8,
  "universalism": 0.7
}
```
