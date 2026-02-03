<div align="center">

<h1>pydantic-collab</h1>

**Build AI agent teams that collaborate intelligently — with handoffs, consultations, and shared memory.**


[![PyPI version](https://img.shields.io/pypi/v/pydantic-collab)](https://pypi.org/project/pydantic-collab/)
[![Tests](https://img.shields.io/github/actions/workflow/status/Unfold-Security/pydantic-collab/ci.yml?label=tests)](https://github.com/Unfold-Security/pydantic-collab/actions)
[![Coverage](https://codecov.io/gh/boazkatzir/pydantic-collab/branch/main/graph/badge.svg)](https://codecov.io/gh/boazkatzir/pydantic-collab)
[![License](https://img.shields.io/pypi/l/pydantic-collab)](https://github.com/Unfold-Security/pydantic-collab/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>


## Features

- 🤝 **Tool calls & handoffs** — Agents consult each other or transfer control
- 🕸️ **Pre-built topologies** — Pipeline, Star, Mesh, or fully custom graphs
- 🧠 **Shared agent memory** — Persistent context across agents during a run
- 🗺️ **Topology visualization** — See your agent graph as an image
- 🐍 **Pydantic-AI Native** — Use Pydantic-AI and Logfire observability
- ⚙️ **Configurable context passing** — Control what flows between agents

##  Installation

```bash
pip install pydantic-collab
```

## 🚀 Quick Start

```python
from pydantic_collab import PipelineCollab, CollabAgent

collab = PipelineCollab(
    agents=[
        CollabAgent(name="Triager", system_prompt="Classify the support ticket by product area and urgency (P0-P3). Extract the core issue."),
        CollabAgent(name="Responder", system_prompt="Draft a helpful response. Acknowledge the issue, provide next steps or workarounds."),
    ],
    model="anthropic:claude-sonnet-4-5",
)

result = collab.run_sync("User email: 'I can't export my data to CSV, the button just spins forever. I need this for a board meeting tomorrow!'")
print(result.output)
```

## When to Use This

Use **pydantic-collab** when you need multiple specialized agents working together, without the need of human
intervention.

**Relevant Use Cases:**
- Multi-stage workflows (triage → investigate → respond)
- Specialist teams (coordinator + domain experts)
- Tasks requiring different perspectives (engineering + legal + security)
- Iterative refinement loops (drafter ↔ critic)

## ⚒️  🤝 Tool Calls vs Handoffs

| | Tool Calls (`agent_calls`) | Handoffs (`agent_handoffs`) |
|---|---|---|
| **Purpose** | Get help, stay in control | Transfer control completely |
| **After call** | Caller continues | Caller stops |
| **Use when** | Help me with X | Take over from here |

## Common Topologies

### Pipeline (Sequential)

```python
from pydantic_collab import PipelineCollab, CollabAgent

# Incident postmortem pipeline
collab = PipelineCollab(
    agents=[
        CollabAgent(name="TimelineBuilder", system_prompt="Construct a timeline from the incident channel logs. List each event with timestamp and actor."),
        CollabAgent(name="RootCauseAnalyzer", system_prompt="Identify contributing factors and the root cause. Distinguish symptoms from causes."),
        CollabAgent(name="ActionItemWriter", system_prompt="Propose concrete, assignable action items with owners. Prioritize by impact."),
    ],
    model="openai:gpt-5.2-pro"
)

# Use: collab.run_sync(slack_channel_export)
```

### Star (Hub & Spoke)

```python
from pydantic_collab import StarCollab, CollabAgent

# Feature spec review - PM coordinates specialists
collab = StarCollab(
    agents=[
        CollabAgent(name="ProductLead", system_prompt="Coordinate the review. Synthesize feedback from specialists into a go/no-go recommendation."),
        CollabAgent(name="EngineeringReviewer", system_prompt="Assess technical feasibility, estimate complexity, flag architectural concerns."),
        CollabAgent(name="DesignReviewer", system_prompt="Evaluate UX implications, accessibility, consistency with design system."),
        CollabAgent(name="SupportReviewer", system_prompt="Predict support burden, documentation needs, and customer confusion risks."),
    ],
    model="openai:gpt-5.2-pro"
)

# ProductLead consults each specialist, then synthesizes
```

### Mesh (Everyone talks to everyone)

```python
from pydantic_collab import MeshCollab, CollabAgent

# Vendor evaluation - each perspective needs input from others
collab = MeshCollab(
    agents=[
        CollabAgent(name="SecurityReviewer", system_prompt="Assess security posture: SOC2, data handling, access controls. Consult Legal on compliance implications."),
        CollabAgent(name="EngineeringEvaluator", system_prompt="Evaluate integration effort, API quality, scalability. Consult Security on auth requirements."),
        CollabAgent(name="LegalReviewer", system_prompt="Review contract terms, data processing agreements, liability. Consult Security on data residency."),
    ],
    model="openai:gpt-5.2-pro"
)

# Each agent can consult others - security implications affect legal review, etc.
# Use: collab.run_sync("Evaluate Acme Corp's proposal for our analytics pipeline: ...")
```

<details>
<summary><b>Custom Topology</b></summary>

Define explicit tool calls and handoffs:

```python
from pydantic_collab import Collab, CollabAgent

# Code review with specialist consultation
collab = Collab(
    agents=[
        CollabAgent(
            name="LeadReviewer",
            system_prompt="Review the PR for correctness and design. Consult specialists for deep dives.",
            agent_calls=["SecurityChecker", "PerfAnalyzer"],  # Can consult
            agent_handoffs="SummaryWriter",                   # Hands off for final summary
        ),
        CollabAgent(name="SecurityChecker", system_prompt="Check for vulnerabilities: injection, auth issues, secrets."),
        CollabAgent(name="PerfAnalyzer", system_prompt="Identify performance issues: N+1 queries, blocking calls, memory."),
        CollabAgent(name="SummaryWriter", system_prompt="Compile all feedback into a clear, actionable review summary."),
    ],
    model="openai:gpt-5.2-pro",
   final_agent="SummaryWriter"
)
```

</details>

## 🧠 Agent Memory

Share persistent context between agents during a run:

```python
from pydantic_collab import PipelineCollab, CollabAgent, AgentMemory

incident_context = AgentMemory(
    name="incident",
    description="Accumulated incident context: affected systems, customer impact, timeline"
)

collab = PipelineCollab(
    agents=[
        CollabAgent(
            name="FirstResponder",
            system_prompt="Gather initial context: what's broken, who's affected, when it started. Document in memory.",
            memory={incident_context: "rw"},  # Writes initial context
        ),
        CollabAgent(
            name="Investigator",
            system_prompt="Deep dive into the root cause using the established context. Add findings to memory.",
            memory={incident_context: "rw"},  # Reads context, adds findings
        ),
        CollabAgent(
            name="CommunicationDrafter",
            system_prompt="Draft customer and internal communications based on the full incident context.",
            memory={incident_context: "r"},   # Reads accumulated context
        ),
    ],
    model="openai:gpt-5.2-pro"
)
```

<details>
<summary><b>AgentMemory syntax</b></summary>

```python
# Simple string (defaults to 'rw')
CollabAgent(name="Agent", memory="notes")

# List (all default to 'rw')
CollabAgent(name="Agent", memory=["notes", "decisions"])

# Dict for explicit permissions
CollabAgent(name="Agent", memory={"notes": "rw", "config": "r"})
```

</details>

## Visualizing Topology

```python
collab = Collab(...)
collab.visualize_topology()  # Opens image
collab.visualize_topology(save_path="topology.png", show=False)  # Save to file
```

```bash
pip install pydantic-collab[viz]  # Requires visualization dependencies
```

![Topology example](docs/topology_example.png)

## Adding Tools

```python
collab = Collab(agents=[...], model="openai:gpt-4o-mini")

@collab.tool_plain
async def query_metrics(service: str, time_range: str) -> str:
    """Query Prometheus metrics for a service."""
    # ... integration with your metrics system
    return metrics_data

@collab.tool_plain(agents=("Investigator",))  # Only for specific agents
async def get_recent_deploys(service: str) -> str:
    """Get recent deployments for a service."""
    return deployment_history
```

## Result Object

```python
result = collab.run_sync("Query")

result.output              # Final output
result.final_agent         # Agent that produced output
result.execution_path      # ["Triager", "Investigator", "Responder"]
result.usage               # Token usage statistics

print(result.print_execution_flow())  # Visual flow diagram
```

## Configuration

<details>
<summary><b>Execution Limits</b></summary>

```python
collab = Collab(
    agents=[...],
    max_handoffs=10,           # Maximum handoff iterations (default: 10)
    max_agent_call_depth=3,    # Maximum recursive tool call depth (default: 3)
)
```

</details>

<details>
<summary><b>Handoff Settings</b></summary>

Control what information flows between agents during handoffs:

```python
from pydantic_collab import CollabSettings

collab = Collab(
    agents=[...],
    collab_settings=CollabSettings(
        include_conversation="allow",      # "allow", "disallow", "force"
        include_thinking="disallow",
        include_handoff="allow",
        include_topology_in_prompt=True,
    ),
)
```

</details>

<details>
<summary><b>Custom Prompt Builder</b></summary>

```python
from pydantic_collab import PromptBuilderContext, CollabSettings

def my_prompt_builder(ctx: PromptBuilderContext) -> str:
    lines = [f"Agent: {ctx.agent.name}"]
    if ctx.can_handoff:
        lines.append(f"Hand off to: {', '.join(a.name for a in ctx.handoff_agents)}")
    return "\n".join(lines)

collab = Collab(
    agents=[...],
    collab_settings=CollabSettings(prompt_builder=my_prompt_builder),
)
```

</details>

<details>
<summary><b>Using Dependencies</b></summary>

```python
from pydantic import BaseModel

class MyDeps(BaseModel):
    db: Database
    cache: Cache

collab = Collab(agents=[...])
result = collab.run_sync("...", deps=MyDeps(db=db, cache=cache))
```

</details>

## Examples

See [`examples/`](examples/) for complete working examples:

| Example | Description |
|---------|-------------|
| [`01_simple_chain.py`](examples/01_simple_chain.py) | Basic forward handoff pipeline |
| [`02_bidirectional_chain.py`](examples/02_bidirectional_chain.py) | Agents can handoff back |
| [`04_mesh_network.py`](examples/04_mesh_network.py) | Full mesh collaboration |
| [`08_mesh_with_tools.py`](examples/08_mesh_with_tools.py) | Mesh with function tools |
| [`12_data_analysis_pipeline.py`](examples/12_data_analysis_pipeline.py) | Complex multi-stage workflow |

```bash
uv run --env-file .env examples/01_simple_chain.py
```


## License 

This project is licensed under the MIT License - see the LICENSE file for details.
