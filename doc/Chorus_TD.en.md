# Chorus — Technical Design

---

## 1. Core Mechanism Choices

Three key LangGraph mechanism decisions before the graph structure:

**Parallel evaluation via the `Send` API:** 7 agents run in parallel, not sequentially. LangGraph's `Send` dynamically creates branches at runtime; results are merged using `Annotated` + reducer to avoid the anchoring bias caused by sequential execution.

**Entropy routing via `add_conditional_edges`:** The Entropy Monitor is a conditional routing node that decides — based on entropy score and current debate round — whether to proceed to consensus or enter debate.

**Debate loop as a back-edge:** `debate_node → entropy_monitor_node` forms a directed cycle, natively supported by LangGraph without special syntax.

---

## 2. Graph Structure

```
START
  │
  ▼
intake_node
  │  load value_vector, initialize control fields
  ▼
decision_classifier_node
  │  LLM classifies to fixed enum, loads preset weight template by type
  ▼
bias_detection_node
  │  LLM detects cognitive biases, generates bias_flags metadata
  ▼
reality_node
  │  entity extraction → web search → gap detection, outputs reality_discrepancies
  │
  ├─(conditional edge: dispatch_node)─►  agent_node("Arbiter")    ──┐
  │                                  ►  agent_node("Empath")     ──┤
  │                                  ►  agent_node("Soothsayer") ──┤
  │                                  ►  agent_node("Compass")    ──┤ results merged
  │                                  ►  agent_node("Narrator")   ──┤ (reducer)
  │                                  ►  agent_node("Conscience") ──┤
  │                                  ►  agent_node("Guardian")   ──┘
  │                                                               │
  │                                                               ▼
  │                                                     entropy_monitor_node
  │                                                       │  compute divergence, identify conflict pairs
  │                                                       │
  │                                  ┌────────────────────┼──────────────────┐
  │                                  │                    │                  │
  │                           converged branch      high-entropy branch   irreconcilable branch
  │                    (low entropy / rounds       rounds < max          (multi_polar /
  │                     exceeded / no entropy                             irreconcilable)
  │                     drop / oscillation)
  │                                  │                    │                  │
  │                                  │              debate_node              │
  │                                  │        (select most contested         │
  │                                  │         option; pick most             │
  │                                  │         opposed agent pair;           │
  │                                  │         debate_round += 1)            │
  │                                  │                    │                  │
  │                                  │                    └─── back-edge ────┤
  │                                  │                         → entropy     │
  │                                  ▼                                       │
  │                            consensus_node  ◄──────────────────────────────┘
  │                              │  per-option weighted aggregation, output option ranking
  │                              ▼
  │                        persona_updater_node
  │                              │  write decision_history (SQLite)
  │                              │  generate markdown report to data/reports/
  │                              ▼
  │                             END
```

---

## 3. State Design

| Field | Type | Description |
|-------|------|-------------|
| `username` | str | User identifier |
| `user_narrative` | str | User's decision narrative |
| `decision_options` | list[str] | Candidate options |
| `value_vector` | dict[str, float] | Schwartz 10-dimension weights |
| `decision_type` | str | Decision type enum |
| `scene_template` | dict[str, float] | Agent scene-weight template |
| `bias_flags` | list[dict] | Cognitive bias metadata |
| `reality_discrepancies` | list[dict] | Web-search findings; each entry contains `entity`, `narrative_claim`, `reality_fact`, `severity` (low/medium/high) |
| `agent_stances` | Annotated[dict, or_] | Each agent's evaluation result; merged across parallel branches with `operator.or_` |
| `initial_stances` | dict | Phase 1 stance snapshot; written once by `entropy_monitor_node` when `debate_round == 0`; read-only thereafter |
| `critical_agents` | list[str] | Agents whose failure crashes the entire graph |
| `failed_agents` | list[str] | Agents detected missing by the health check |
| `entropy_score` | float | Current round entropy (weighted mean std dev across options, weights from value_vector) |
| `last_entropy_score` | float \| None | Previous round entropy, used for oscillation detection |
| `conflict_type` | str | `binary` / `outlier` / `multi_polar` / `converged` |
| `conflicting_agents` | list[str] | Agents participating in debate |
| `debate_round` | int | Current debate round |
| `max_debate_rounds` | int | Hard cap, default 3 |
| `stance_history` | Annotated[list, add] | Per-round snapshot of each agent's `option_scores`, used for oscillation detection |
| `debate_history` | Annotated[list, add] | Full debate transcript, appended each round |
| `consensus` | dict | Weighted aggregation result (`option_scores` + `top_option` + `weights`) |
| `antagonism_flags` | list[str] | Antagonistic agent annotations |
| `final_recommendation` | str | LLM-generated final recommendation text |
| `final_report` | str | Full markdown report (structured data + LLM recommendation) |

`AgentStance` structure: `option_scores` (per-option score, -1.0 to 1.0), `reasoning` (rationale, ≤60 chars), `confidence` (0.0 to 1.0).

---

## 4. Key Node Logic

### intake_node

Loads `value_vector` from SQLite. On first use, calls `interrupt()` to pause and wait for the user to complete the Schwartz 10-dimension scoring before writing to the DB and continuing. Initializes all control fields: `debate_round=0`, `max_debate_rounds=3`, `critical_agents=["Arbiter","Empath"]`, etc.

### decision_classifier_node

**7 decision types** (derived from 7 psychological dimensions, each type has 1–2 lead dimensions):

| Type | Description | Lead Agents |
|------|-------------|-------------|
| `career` | Career & profession | Arbiter, Compass |
| `finance` | Finances & assets | Arbiter, Conscience |
| `relationship` | Intimate relationships & family | Empath, Guardian |
| `relocation` | Residence & migration | Guardian, Soothsayer |
| `health` | Health & body | Soothsayer, Empath |
| `identity` | Identity & personal growth | Narrator, Compass |
| `ethics` | Ethics & social responsibility | Conscience, Compass |

LLM classifies `user_narrative` to the above fixed enum at `temperature=0` (Pydantic `Literal` constraint). Loads the preset `scene_template` from the DB by type.

### bias_detection_node

LLM identifies cognitive biases (13 types) in the narrative at `temperature=0`, generating `bias_flags`. Each flag contains: bias name, agent most needing to be aware of it, description specific to this narrative (≤40 chars). Only surfaces phenomena — no value judgments.

### reality_node

Three-step pipeline anchoring the user narrative to verifiable external facts:

```
Step 1 — Entity extraction (LLM, temperature=0)
    Extract searchable entities from user_narrative
    Criterion: searching the entity's latest information would be useful for the decision
               (includes unverified user claims AND important context the user didn't mention)
    Output: entities: list[str]

Step 2 — Web search (Tavily, parallel per entity)
    Call AsyncTavilyClient.get_search_context() for each entity
    Store results as local variable reality_context: {entity: search_summary}
    (not written to State)

Step 3 — Gap detection (LLM, temperature=0)
    Compare user_narrative with reality_context; output findings relevant to the decision
    Flag both: user statement differs from reality / search reveals important background not mentioned
    Output reality_discrepancies: list[{
        entity: str,
        narrative_claim: str,    # user's words, or "not mentioned"
        reality_fact: str,       # verified fact or background info
        severity: "low"|"medium"|"high"
    }]
```

Degradation: if `TAVILY_API_KEY` is absent, no entities are extractable, or a search error occurs, returns `reality_discrepancies = []` and downstream nodes continue in "no reality data" mode.

### dispatch_node (conditional edge routing function)

Not a node — a routing function for `reality_node`'s conditional edge. Returns `list[Send]`, dynamically creating a parallel branch for each `agent_name`:

```
Return Send list: for each agent_name in AGENT_NAMES
    Send("agent_node", state + {agent_name})
```

### agent_node

7 agents share one node function, differentiated by `agent_name` to select the prompt template. **Independently scores each candidate option** (-1.0 to +1.0). Other agents' stances are not passed in, guaranteeing independent evaluation free from anchoring.

```
try:
    Use LLM structured output to generate AgentStance
      - option_scores: {option_name: score} covering all candidate options
      - reasoning: ≤60 chars, minimal jargon
      - confidence: 0.0–1.0
    Context includes: user_narrative + bias_flags + value_vector + decision_options
                    + reality_discrepancies
    Return {agent_stances: {agent_name: stance}}
except:
    if agent_name in critical_agents → raise (crash the graph)
    else → return {} (silent skip; entropy_monitor fills with neutral values)
```

Node is mounted with `RetryPolicy(max_attempts=3)`.

### entropy_monitor_node

Three steps: health check → compute entropy and classify conflict → record snapshot.

```
Health check:
    missing = AGENT_NAMES - agent_stances.keys()
    if missing contains critical_agents → flag SYSTEM_PARTIAL_FAILURE
    if missing contains non-critical agents → fill with option_scores all 0.0

if debate_round == 0 → snapshot current agent_stances into initial_stances (read-only thereafter)

Compute:
    # Value-aware entropy: agents weighted by their value_vector mean,
    # amplifying divergence in dimensions the user actually cares about
    agent_weight[a] = avg(value_vector[d] for d in AGENT_VALUE_MAPPING[a])
    per-option weighted std dev → entropy = mean across all options

    conflict_type, conflicting_agents = classify_conflict(stances)
        Classification order: converged → outlier → binary → multi_polar

        converged   : max(per-option std dev across agents) < SCORE_RANGE * 0.05
                      Uses per-option std (not agent-mean std) to prevent
                      "compensatory preference" being misclassified as convergence
                      (e.g., A prefers X and rejects Y; B prefers Y and rejects X —
                      means are similar but actual divergence is high)

        outlier     : find the agent with the largest mean deviation (outlier_candidate);
                      compute std_dev of remaining agents (rest_std);
                      if rest_std / total_std < 0.5 → outlier
                      (that agent accounts for more than half of total scatter)

        binary      : dynamically split camps by median of agent means, ≥ 2 each
                      no fixed threshold; works correctly for all-positive or all-negative distributions

        multi_polar : all other cases
    if conflict_type == multi_polar → set antagonism_flags

Snapshot: append current {agent: option_scores} to stance_history
```

### Conditional routing function: route_after_entropy

```
if conflict_type in (multi_polar, converged)          → consensus_node
if debate_round >= max_debate_rounds                  → consensus_node
if entropy < threshold                                → consensus_node  (converged)
if debate_round > 0 and |entropy - last_entropy| < ε → consensus_node  (no entropy drop)
if debate_round > 0 and flatten(stance[N]) ≈ flatten(stance[N-2]) → consensus_node  (oscillation)
else                                                  → debate_node
```

The threshold is coupled to `value_vector` (high conservation dimensions → lower threshold; high openness dimensions → higher threshold). Oscillation detection flattens `stance_history` into `{agent:option: score}` vectors and compares L1 distance.

### debate_node

Back-edge origin. Representative selection logic:

```
1. Find the most contested option: max(options, key=option_std_dev across all agents)

2. binary:
   On that option, sort conflicting_agents by score and split at the median:
   - Top camp: highest scorer → rep_a; rest → ally_a
   - Bottom camp: lowest scorer → rep_b; rest → ally_b
   (no fixed threshold; works for all-positive or all-negative distributions)

3. outlier:
   On that option:
   - Agent farthest from the group mean → rep_a
   - Agent closest to the group mean → rep_b
   - No allies passed (prevents majority pressure)
```

Debate execution:

```
debate_context = compressed debate_history (old rounds summarized + latest round verbatim)
rep_a new stance = LLM(prompt + initial_stance + my_stance + opponent + allies + context)
rep_b new stance = LLM(prompt + initial_stance + my_stance + opponent + allies + context)
Return updated agent_stances, debate_round+1, appended debate_history (with reason field)
```

`bias_flags` are presented as "potential cognitive blind spots in the user's narrative," prompting agents to actively examine them — not converted into attack instructions.

### consensus_node

```
Two-layer weight computation:
    value_score[a]  = avg(value_vector[d] for d in AGENT_VALUE_MAPPING[a])
    raw_weight[a]   = scene_template[a] × value_score[a]
    weights[a]      = raw_weight[a] / sum(raw_weight)   # normalized so Σweights = 1

Per-option weighted score:
    option_scores[opt] = Σ agent_stances[a].option_scores[opt] × weights[a]

top_option = argmax(option_scores)
ranked_options = sorted(option_scores, descending)

Call LLM to generate final recommendation (≤300 words, minimal jargon)
```

**Agent–Schwartz mapping:**

| Agent | Schwartz Dimensions |
|-------|-------------------|
| Arbiter | achievement, self_direction |
| Empath | hedonism, stimulation |
| Soothsayer | security, conformity |
| Compass | universalism, self_direction |
| Narrator | self_direction, benevolence |
| Conscience | universalism, benevolence, tradition |
| Guardian | benevolence, security |

### persona_updater_node

Two responsibilities:

**1. Write to SQLite:** For each agent, compute alignment using the `top_option` score:
```
alignment = 1 - |agent_score[top_option] - consensus_score[top_option]| / 2
```

**2. Generate markdown report:** Prepare a data dict from state, render `templates/report.md.j2` with Jinja2, and save to `data/reports/<username>_<timestamp>.md`.

Report contains: decision background, cognitive biases, per-dimension × per-option score matrix (with before/after debate comparison), option ranking, debate transcript (with selection rationale), final recommendation, and dimension weight breakdown.

---

## 5. Graph Construction

```
Node registration:
    intake, decision_classifier, bias_detection, reality,
    agent_node (RetryPolicy max=3),
    entropy_monitor, debate, consensus, persona_updater

Static edges:
    START → intake → decision_classifier → bias_detection → reality
    agent_node → entropy_monitor
    debate → entropy_monitor  (back-edge)
    consensus → persona_updater → END

Conditional edges:
    reality_node --dispatch_node--> ["agent_node"]
    entropy_monitor_node --route_after_entropy--> ["debate_node", "consensus_node"]

Compilation: no custom Checkpointer (LangGraph API platform manages persistence)
```

---

## 6. Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Framework | LangGraph (Python) | Supports cycles, Annotated State, Send API |
| LLM | Qwen (DashScope) | Connected via `langchain_openai.ChatOpenAI` + DashScope-compatible endpoint; model name configured via `LLM_MODEL` env var |
| Storage | SQLite | Decision history (Evolving Persona) + scene template registry |
| Report template | Jinja2 | `templates/report.md.j2`; data preparation and formatting fully separated |
| Debugging | LangGraph Studio | Visualize node flow and state changes |

---

## 7. Design Notes

**LLM Context Management:**

| Node | Call Type | Context Source |
|------|-----------|----------------|
| decision_classifier_node | Single-turn | user_narrative; temperature=0, classification only |
| bias_detection_node | Single-turn | user_narrative; temperature=0, bias detection only |
| reality_node | Single-turn × 3 steps | Step1: user_narrative → entities; Step2: Tavily parallel search (1 call/entity, results as local var); Step3: narrative + search summaries → reality_discrepancies; all temperature=0 |
| agent_node (Phase 1) | Single-turn | user_narrative + bias_flags + value_vector + decision_options + reality_discrepancies; other agents' stances isolated |
| debate_node | Multi-turn | my_stance + opponent + ally_stances + debate_context (compressed view); non-debating agents' stances isolated |
| consensus_node | Single-turn | all agent_stances + option_scores + ranked_options + debate_history |

**Multi-option scoring design:** Each agent independently scores all candidate options, avoiding collapse of multi-option decisions into a binary stance. Entropy is computed as the mean per-option cross-agent standard deviation; consensus weighting aggregates each option separately, outputting a ranked list rather than a single recommendation score.

**Debate representative selection:** First find the most contested option (highest cross-agent standard deviation), then select the two most opposed agents on that option. The binary and outlier conflict types share the same selection logic; the difference is whether allies are passed in.

**agent_node reuse:** 7 agents share one node function, differentiated by the `agent_name` parameter selecting the prompt template. Adding a new agent requires no graph structure changes.

**dispatch_node is a routing function, not a node:** A function returning `list[Send]` cannot be used as a regular node; it must be registered as the routing function of `add_conditional_edges` and declare its target node list for LangGraph's static analysis.

**Single DB, two tables:** `persona_updater_node`'s DAO writes to the `decision_history` table; the scene template registry uses a separate table in the same `chorus.db`, with no cross-table interference.

**Maximum debate rounds:** `max_debate_rounds` defaults to 3 and is configurable at initialization. When exceeded, the consensus report explicitly annotates antagonistic dimensions and does not conceal conflicts.

**Parallel result aggregation with degradation:** Two layers. Layer 1: `RetryPolicy(max_attempts=3)` handles transient API failures. Layer 2: critical agents (default Arbiter + Empath) crash the entire graph after retries are exhausted; non-critical agents are silently skipped, and `entropy_monitor_node` fills in all-zero `option_scores` and continues.

**Oscillation detection:** `route_after_entropy` has five exit conditions: multi_polar/converged, round cap, low-entropy convergence, no entropy drop, and periodic oscillation. The last two conditions only apply when `debate_round > 0` — on round 0, `last_entropy_score` may carry a residual value from the previous graph execution; without the guard, it would mistakenly trigger, causing debate to be skipped before it starts. Oscillation detection flattens `stance_history` into `{agent:option: score}` vectors and compares L1 distance between round N and round N-2; symmetric oscillation cannot be detected by entropy difference alone and requires vector comparison.

**initial_stances anchor:** `operator.or_` overwrites debating agents' stances after each round, causing Phase 1 original conclusions to be lost. `entropy_monitor_node` snapshots the stances into `initial_stances` when `debate_round == 0`; read-only thereafter. The report's scoring matrix shows the "before→after" format.

**debate_history compression:** The complete original record is always preserved in State. What is passed to the LLM is a compressed view: when there is more than 1 round, older rounds are summarized and only the latest round is kept verbatim, preventing the context window from growing with each round.

**Report generation:** `persona_updater_node`'s `_prepare_data()` handles data preparation; `_render_report()` handles Jinja2 rendering — the two layers are completely separated. The template file `templates/report.md.j2` can be modified independently without touching Python logic.
