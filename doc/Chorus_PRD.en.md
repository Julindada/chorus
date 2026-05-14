# Chorus
> A psychology-driven multi-agent decision support system built on LangGraph

---

## 1. Core Concept

Chorus is a Multi-Agent Decision Support System for navigating complex decisions. Rather than offering advice from the outside, it **simulates the internal conflict and integration process of human psychology** — multiple agents representing different psychological dimensions engage in cyclic debate and ultimately produce a consensus-audited decision report.

**Core assumption:** Human decision errors are usually not caused by a lack of information, but by one psychological dimension (e.g., short-term emotion or habitual avoidance) suppressing all others. Chorus's goal is to ensure every dimension is genuinely heard, then find the optimal path through reasoned integration.

**Technical analogy:** The entire decision process is a **state machine**. Each option (e.g., a job offer) is the input. The system filters and evaluates it through 7 psychological nodes, triggers loops until consensus is reached, and outputs a report.

---

## 2. Psychological Foundations

### 2.1 Dual Process Theory

Kahneman's dual process theory is the underlying framework: the conflict between System 1 (automatic, emotion-driven) and System 2 (deliberate, analytic) is the root of most decision dilemmas. Empath and Soothsayer represent System 1; Arbiter represents System 2; the remaining four agents represent deeper value- and identity-driven layers.

### 2.2 Theory of Basic Human Values (Schwartz)

The Schwartz model arranges 10 basic values — Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, Universalism — on a circular continuum. Adjacent values are compatible; opposing values are in natural tension (e.g., Self-Direction ↔ Conformity).

**Engineering implementation:** The user profile stores `value_vector: dict[str, float]` (each dimension 0.0–1.0), initialized during onboarding. The Conscience agent uses this vector directly for deviation analysis; consensus weighting across all agents also references it.

---

## 3. Agent Personas

The system has 7 agents, each corresponding to a distinct psychological dimension.

| Dimension | Agent Name | Psychological Basis | Core Function |
|-----------|------------|---------------------|---------------|
| Logical Reasoning | **Arbiter** | System 2 / Prefrontal Cortex | Cost-benefit analysis, probability judgment, causal reasoning |
| Emotional Intuition | **Empath** | System 1 / Appraisal Theory | Surface hidden emotional signals; identify fear, excitement, aversion |
| Somatic Warning | **Soothsayer** | Somatic Marker Hypothesis (Damasio) | Predict decision impact on stress level, energy, and sleep |
| Meaning & Purpose | **Compass** | Logotherapy (Frankl) | Assess whether the decision aligns with the user's life narrative and long-term goals |
| Self & Identity | **Narrator** | Narrative Identity Theory (McAdams) | Detect whether the decision conflicts with "who I believe I am" |
| Moral Calibration | **Conscience** | Schwartz Value Vector | Faithfully present the gap between the decision and the user's value vector — report only, no judgment |
| Relational Impact | **Guardian** | Attachment Theory | Assess impact on family, partner, and core social circle |

> **Design principle:** Conscience's role is to "faithfully present," not to "adjudicate" — it states the deviation but does not rule on right or wrong. Final weighing is left to the consensus mechanism. This prevents the moral dimension from dominating others.

### 3.1 Natural Tension Pairs

High-conflict pairs that are prioritized in the debate phase:

| Tension Pair | Psychological Reason |
|-------------|---------------------|
| Arbiter ↔ Empath | System 2 vs System 1 — the most classic conflict |
| Arbiter ↔ Soothsayer | Rational gain vs bodily cost |
| Compass ↔ Arbiter | Meaning cannot be captured by a utility function |
| Narrator ↔ Guardian | Self-actualization vs impact on others |

---

## 4. System Architecture

### 4.1 Contextual Dynamic Weighting

**Problem:** A static `value_vector` cannot distinguish priority differences across decision types. "Career change" and "family relocation" have very different importance distributions across agents.

**Mechanism:** The Decision Classifier node categorizes the narrative into one of 7 fixed decision types, then loads the corresponding preset scene-weight template.

**7 built-in decision types:**

| Type | Description | Lead Agents | Low-weight Agents |
|------|-------------|-------------|-------------------|
| career | Career & profession | Arbiter, Compass | Guardian |
| finance | Finances & assets | Arbiter, Conscience | Empath |
| relationship | Intimate relationships & family | Empath, Guardian | Arbiter |
| relocation | Residence & migration | Guardian, Soothsayer | Empath |
| health | Health & body | Soothsayer, Empath | Arbiter |
| identity | Identity & personal growth | Narrator, Compass | Conscience |
| ethics | Ethics & social responsibility | Conscience, Compass | Soothsayer |

These 7 types cover the vast majority of major life decisions. The type set is fixed to ensure classification consistency.

**Final weights:** The scene-weight template is multiplied element-wise with the user's Schwartz value vector and then normalized — reflecting both the objective characteristics of the decision type and the user's personal value priorities.

### 4.2 Cognitive Bias Middleware

**Mechanism:** An independent LLM call (`temperature=0`) runs after decision classification and before the psychological nodes. It identifies cognitive biases in the narrative and outputs a structured `list[BiasFlag]`, fully isolated from the 7 downstream agents.

13 cognitive biases are currently detected:

| Bias | Typical Manifestation | Primary Recipients |
|------|-----------------------|-------------------|
| Sunk Cost Fallacy | Hard to abandon due to time/money already invested | Arbiter |
| Recency Effect | Recent events dominate the overall judgment | Empath |
| Bandwagon Effect | Others' choices as the primary basis | Narrator |
| Black-and-White Thinking | Either/or framing; middle paths ignored | Arbiter |
| Confirmation Bias | Severe imbalance of positive/negative evidence | All agents |
| Loss Aversion | Fear of loss far exceeds expected equivalent gain | Arbiter, Empath |
| Status Quo Bias | Change itself perceived as the risk | Compass, Narrator |
| Anchoring Effect | Overly anchored to a specific number or date | Arbiter |
| Overconfidence Bias | Optimism exceeds actual supporting evidence | Soothsayer, Conscience |
| Emotional Reasoning | Current emotion used as evidence of fact | Arbiter |
| Planning Fallacy | Underestimating time/resources required | Soothsayer, Arbiter |
| Catastrophizing | Imagining negative outcomes at their extreme worst | Empath, Soothsayer |
| Should Statements | Moral obligation substitutes for exploring true desires | Conscience, Narrator |

**Design principle:** Only flags are surfaced — no value judgments. Bias annotations are passed as metadata to the relevant agents, who decide independently how to weight them; there is no forced intervention in conclusions.

### 4.3 Reality Grounding

**Problem:** All 7 agents work in the subjective layer — analyzing the user's emotions, values, somatic signals, and identity. No node verifies whether the user's description of external reality is accurate. If the user's perception of the outside world is distorted, the psychological analysis rests on a false foundation.

**Position in the graph:** After bias detection, before parallel dispatch. Running bias detection first lets reality grounding know where the user's cognition may be distorted, enabling targeted supplementation. All 7 agents share the same reality baseline, maintaining information symmetry.

**Output: `reality_discrepancies`**

Extractable entities are pulled from the narrative, web-searched via Tavily for objective information, and integrated into a structured findings list passed to all agents. Each finding contains: entity, user's original statement (or "not mentioned"), verified fact, severity (low / medium / high).

Both cases are flagged: discrepancies between what the user said and reality, and important background information the user didn't mention but that is relevant to the decision.

**Dual nature:** A cognitive gap is simultaneously a factual signal (updating agents' information base) and a psychological signal (the user's perception itself is meaningful). Each agent interprets it independently through its own psychological lens. Chorus does not use external facts to override internal feelings.

**Triggering and degradation:** Triggered when there are searchable entities in the narrative. If `TAVILY_API_KEY` is absent, no entities are extractable, or a search error occurs, `reality_discrepancies` is empty and the graph continues without blocking.

### 4.4 Cycling & Debate Mechanism

Unlike a traditional linear chain, Chorus uses LangGraph's **cycles** to implement multi-round debate.

```
User inputs decision options
        ↓
  [Load user profile] ── read Schwartz value vector
        ↓
  [Decision classification] ── identify decision type, load scene weights
        ↓
  [Cognitive bias detection] ── identify biases, generate prompt flags
        ↓
  [Reality grounding] ── extract external facts, detect narrative gaps
        ↓
  7 agents evaluate independently and simultaneously
  (isolated from each other to prevent anchoring)
  Arbiter / Empath / Soothsayer / Compass / Narrator / Conscience / Guardian
        ↓
  [Entropy monitor] ── compute opinion divergence, identify conflict structure
        │
        ├─ converged ──► [Consensus node] → output decision report
        │                                        ↓
        │                                [Profile updater] ← record decision alignment
        │
        └─ divergent ──► [Debate node] ──► back to entropy monitor (loop)
```

**Parallel independent evaluation:** 7 agents launch simultaneously and cannot see each other's conclusions. This is intentional — preventing the first agent to conclude from anchoring others' judgments, ensuring each psychological dimension gives a genuine independent voice.

**Multi-option scoring:** Each agent independently scores each candidate option (`option_scores: dict[str, float]`, range -1.0 to +1.0) rather than outputting a single stance value. This ensures that in multi-option decisions (e.g., choosing among three cities) every option is fully evaluated and not collapsed into a binary support/oppose signal.

**Three conflict structures:** The entropy monitor identifies the shape of conflict before entering debate:
- **Binary:** Opinion spread exceeds threshold; a dynamic median split creates two camps with ≥ 2 agents each
- **Outlier:** Overall tendency is aligned, but one agent's mean deviates by more than 1 standard deviation
- **Multi-polar (divergent):** Opinions scatter in all directions — debate is skipped; the consensus report flags the divergence

**Debate representative selection:** Before entering debate, the system finds the "most contested option" — the candidate with the highest per-agent standard deviation. It then selects two representatives based on score distribution on that option: the agent with the largest deviation as one side, the agent closest to the overall mean as the other. Both debate; remaining agents are assigned as allies based on stance similarity. The selection logic is explained in plain language in the debate section of the report.

**Isolation principle in debate:** Only the conflicting parties participate. Non-conflicting agents' opinions are not passed into the debate, preventing bystander pressure from causing artificial convergence.

**Loop exit conditions:** Two failure modes are prevented — infinite loops and shallow consensus. Normal exit: opinions converge or the maximum debate rounds are reached. Abnormal exit: opinions show no meaningful change across rounds, or oscillate periodically. In either abnormal case, the report explicitly marks an irreconcilable conflict; no forced pseudo-consensus is generated.

**Entropy threshold coupling:** The divergence threshold that triggers debate is linked to the user's value vector. Users with high Security weighting have a lower threshold and enter deeper discussion more easily; users with high Stimulation weighting have greater tolerance for uncertainty and a correspondingly higher threshold.

### 4.5 Evolving Persona

**Problem:** Human values change over time and context. A one-time calibrated `value_vector` cannot capture this evolution.

**Mechanism:** After each decision, `persona_updater_node` converts the difference between each agent's initial score on the **recommended option** and the final composite score into an alignment value (0.0–1.0) stored in the SQLite `decision_history` table. Alignment formula: `1.0 - |agent_score - top_score| / 2.0`.

**Historical pattern display:** At the next intake, the system reads the records and appends statistics to the report using data rather than personality labels:

> Historical reference: across 5 past career decisions, the average alignment between the final decision and Soothsayer's opinion was 0.31 (out of 1.0). It is recommended to pay closer attention to this agent's analysis this time.

Personalizing statements like "you tend to…" are avoided to prevent the labeling effect (repeatedly labeled self-perceptions can solidify into self-fulfilling prophecies). SQLite is used instead of a vector DB (historical records are structurally regular and do not require semantic retrieval).

---

## 5. Output Format

After each decision run, `persona_updater_node` renders a Markdown report using a Jinja2 template and saves it to `data/reports/<username>_<timestamp>.md`. The report includes:

1. **Decision background:** The user's original narrative

2. **Cognitive bias detection (if any):** Structured table listing detected bias types, affected dimensions, and descriptions

3. **Dimension evaluation matrix:** A table with options as columns and agents as rows
   - Each cell shows that agent's score for that option (-1.0 to +1.0)
   - If scores changed after debate, the format is "before→after"
   - Last row: composite weighted score per option
   - Weight column: explains the dimension's weight (determined by decision type template × user value vector)

4. **Dimension arguments:** Each agent explains its score in a paragraph using plain language

5. **Option ranking:** Options sorted by composite score, with recommended option marked

6. **Debate transcript (if any):** Each round's participants, selection rationale (why these two dimensions were chosen to debate), and updated arguments

7. **Final recommendation:** LLM-generated synthesis (≤300 words); if irreconcilable conflict exists, it is presented honestly rather than forced into a single conclusion

8. **Dimension weight breakdown:** Explains the value dimensions behind each agent's weight and how much the user values them
