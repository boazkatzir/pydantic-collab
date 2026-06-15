"""Internal types, settings, and data structures for agent Collabs.

This module contains all type definitions, settings classes, and data structures
used internally by the Collab orchestrator. Users should not import from this module directly.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Generic, Literal, TypeAlias, TypeVar, cast, overload

import pydantic_ai
from pydantic import BaseModel, Field
from pydantic_ai import (
    AbstractToolset,
    EndStrategy,
    InstrumentationSettings,
    ModelSettings,
    RunContext,
    RunUsage,
    Tool,
    ToolsetFunc,
    models,
)
from pydantic_ai._agent_graph import HistoryProcessor
from pydantic_ai.agent import AbstractAgent, AgentInstructions, AgentMetadata, EventStreamHandler, NoneType
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.output import OutputSpec
from pydantic_ai.tools import BuiltinToolFunc, ToolFuncEither, ToolsPrepareFunc
from typing_extensions import TypeAliasType

from pydantic_collab._utils import ensure_tuple

T = TypeVar('T')
# =============================================================================
# Utility Functions (defined here to avoid circular imports)
# =============================================================================


# =============================================================================
# Type Aliases
# =============================================================================
RegularOptions: TypeAlias = Literal['force', 'disallow', 'allow']
t_agent_name: TypeAlias = str
T = TypeVar('T')
AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')
# Can be an agent name, an AbstractAgent instance, or a CollabAgent wrapper
t_agent_desc: TypeAlias = 'str | AbstractAgent | CollabAgent'
t_seq_or_one = TypeAliasType('t_seq_or_one', T | Sequence[T], type_params=(T,))
t_context_name = str

MAXIMUM_MEM_LINE_LENGTH = 8192
# =============================================================================
# Exceptions
# =============================================================================


class CollabError(Exception):
    """Raised when Collab configuration is invalid."""

    pass


# =============================================================================
# Agent Context
# =============================================================================


class AgentContext(BaseModel):
    """Context data for a single agent."""

    messages: list[Any] = Field(default_factory=lambda: cast(list[Any], []))
    last_output: str | None = None


@dataclass
class AgentRunSummary:
    agent_name: str
    start_time: datetime | None = None
    run_time: float | int | None = None
    output: Any = None
    usage: RunUsage | None = None
    messages: list[Any] = field(default_factory=lambda: cast(list[Any], []))

    def __post_init__(self):
        # This is done because usage gets up and up and it's a mutable object. This
        # is not a 100% valid method as it
        # could climb between __init__ and __post_init__,
        # especially with Threads, but it's close enough
        if self.usage is not None:
            self.usage = copy.copy(self.usage)


@dataclass
class CollabState:
    """Dynamic state for a Collab execution.

    Automatically creates per-agent memory based on registered agents.
    """

    query: str
    final_output: str = ''
    agent_contexts: dict[str, AgentContext] = field(default_factory=lambda: cast(dict[str, AgentContext], {}))
    execution_path: list[str] = field(default_factory=lambda: cast(list[str], []))
    execution_history: list[dict[str, Any]] = field(default_factory=lambda: cast(list[dict[str, Any]], []))
    messages: list[AgentRunSummary] = field(default_factory=lambda: cast(list[AgentRunSummary], []))
    memory_objects: dict[AgentMemory, list[str]] = field(default_factory=lambda: defaultdict(list))
    """Used to store memory between or in agent"""

    def get_context(self, agent_name: str) -> AgentContext:
        """Get or create context for an agent."""
        if agent_name not in self.agent_contexts:
            self.agent_contexts[agent_name] = AgentContext()
        return self.agent_contexts[agent_name]

    def record_execution(self, agent_name: str, input_query: str, output: Any):
        """Record an agent execution in history."""
        # Extract output content. Avoid referencing HandOffBase type
        # (defined later) to keep this function lightweight for the type
        # checker — detect a handoff-like object by required attributes.
        if hasattr(output, 'next_agent') and hasattr(output, 'query'):
            output_text = getattr(output, 'query')
            next_agent = getattr(output, 'next_agent', None)
            action = 'handoff'
            reasoning = getattr(output, 'reasoning', '') or ''
        else:
            output_text = str(output)
            next_agent = None
            action = 'final'
            reasoning = ''
        self.execution_history.append(
            {
                'agent': agent_name,
                'action': action,
                'input': input_query[:200] + '...' if len(input_query) > 200 else input_query,
                'output': str(output_text)[:200] + '...' if len(str(output_text)) > 200 else str(output_text),
                'next_agent': next_agent,
                'reasoning': reasoning[:100] + '...' if len(reasoning) > 100 else reasoning,
            }
        )


@dataclass
class CollabRunResult(Generic[OutputDataT]):
    """The final result of an agent run."""

    output: OutputDataT
    """The output data from the agent run."""
    usage: RunUsage
    """Total usage of all agents in this run"""
    iterations: int = 0
    """Number of agent handoffs executed. does not include agent tool call"""
    final_agent: str = ''
    """Name of the agent that produced final output. Right now can be foreseeable,
     in the future possibly not"""
    max_iterations_reached: bool = False
    """Whether the iteration limit was hit and it was stopped unplanned"""

    _state: CollabState = field(repr=False, compare=False, default_factory=lambda: CollabState(query=''))

    def all_messages(self) -> list[AgentRunSummary]:
        """Return all messages."""
        return self._state.messages

    @property
    def execution_path(self) -> list[str]:
        """Get the path of agents that were executed."""
        return self._state.execution_path

    @property
    def execution_history(self) -> list[dict[str, Any]]:
        """Get the full execution history with inputs/outputs."""
        return self._state.execution_history

    def print_execution_flow(self) -> str:
        """Generate a readable execution flow diagram."""
        lines = [
            'Execution Flow:',
            '=' * 50,
        ]

        for i, step in enumerate(self.execution_history):
            prefix = '→ ' if i > 0 else '⚡ '
            agent = step['agent']
            action = step['action']

            lines.append(f'{prefix}{agent}')
            lines.append(f'   Input: {step["input"][:80]}...')
            lines.append(f'   Output: {step["output"][:80]}...')
            if step['reasoning']:
                lines.append(f'   Reason: {step["reasoning"][:60]}...')

            if action == 'handoff':
                lines.append(f'   ↓ routing to {step.get("next_agent", "unknown")}')
            else:
                lines.append('   ✓ FINAL OUTPUT')
            lines.append('')

        lines.append(f'Total iterations: {self.iterations}')
        lines.append(f'Final agent: {self.final_agent}')
        if self.max_iterations_reached:
            lines.append('⚠️  Max iterations reached!')

        return '\n'.join(lines)

    def __str__(self) -> str:
        return str(self.output)


# =============================================================================
# Agent Memory
# =============================================================================


@dataclass(frozen=True)
class AgentMemory:
    name: t_context_name
    """The name of this memory"""
    description: str | None = None
    """LLM-readable description of this memory and when to use it."""


def str_or_am_to_am(mem: AgentMemory | str) -> AgentMemory:
    """Convert a string or AgentMemory to AgentMemory."""
    if isinstance(mem, str):
        return AgentMemory(name=mem)
    return mem


def validate_r_rw(r_or_rw: str) -> Literal['r', 'rw']:
    """Validate that permission is 'r' or 'rw'."""
    if r_or_rw not in ('r', 'rw'):
        raise RuntimeError(f'Value needs to be either r or rw - got {r_or_rw}')
    return r_or_rw


# =============================================================================
# Agent Connection/Wrapper
# =============================================================================


@dataclass(frozen=True)
class CollabAgent:
    """Wrapper for an agent participating in a Collab.

    Defines an agent's role in the Collab including which agents it can
    call as tools and which agents it can hand off to.
    """

    agent: AbstractAgent
    """The underlying pydantic-ai Agent."""

    description: str | None = None
    """Human-readable description of this agent's purpose."""

    agent_calls: Sequence[t_agent_desc] = ()
    """Other agents this agent can call as tools
     (synchronous calls that return to the caller agent).
    Can be agent names (str), AbstractAgent instances, or CollabAgent instances."""

    agent_handoffs: Sequence[t_agent_desc] = ()
    """Other agents this agent can hand off to (transfer control).
    Can be agent names (str), AbstractAgent instances, or CollabAgent instances."""
    name: str | None = None
    """Name of this Collab Agent. Used by other agents for choosing how to hand over"""

    memory: dict[AgentMemory, Literal['r', 'rw']] = field(default_factory=dict)

    @overload
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        description: str | None = None,
        *,
        agent_calls: t_agent_desc | Sequence[t_agent_desc] = (),
        agent_handoffs: t_agent_desc | Sequence[t_agent_desc] = (),
        memory: dict[t_context_name | AgentMemory, Literal['r', 'rw']]
        | Sequence[t_context_name | AgentMemory]
        | None = None,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: AgentInstructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        tool_timeout: float | None = None,
        **_deprecated_kwargs: Any,
    ): ...

    @overload
    def __init__(
        self,
        agent: AbstractAgent,
        description: str | None = None,
        *,
        agent_calls: t_agent_desc | Sequence[t_agent_desc] = (),
        agent_handoffs: t_agent_desc | Sequence[t_agent_desc] = (),
        memory: dict[t_context_name, Literal['r', 'rw']] | Sequence[t_context_name] | None = None,
        name: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        agent: AbstractAgent | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        description: str | None = None,
        *,
        agent_calls: t_agent_desc | Sequence[t_agent_desc] = (),
        agent_handoffs: t_agent_desc | Sequence[t_agent_desc] = (),
        memory: dict[t_context_name, Literal['r', 'rw']] | Sequence[t_context_name] | t_context_name | None = None,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: AgentInstructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        tool_timeout: float | None = None,
        **_deprecated_kwargs: Any,
    ):
        """Create an agent.

        Args:
            agent: The underlying pydantic-ai Agent.
            model: The default model to use for this agent, if not provided,
                you must provide the model when calling it. We allow `str` here since the actual list of allowed models changes frequently.
            description: LLM readable description of this agent. Required for all agents that aren't the starting agent.
            agent_calls: Which other agents can be called
            agent_handoffs: Which other agents can hand off to
            memory: A dictionary, list or instance of AgentMemory or strings (only names) of Context options available
                to the Agents. If supplied as a dictionary, values should be either 'r' or 'rw' - signifying if context
                should be writable using a dedicated tool or only read in the system prompt.
            output_type: The type of the output data, used to validate the data returned by the model,
                defaults to `str`.
            instructions: Instructions to use for this agent, you can also register instructions via a function with
                [`instructions`][pydantic_ai.agent.Agent.instructions] or pass additional, temporary, instructions when executing a run.
            system_prompt: Static system prompts to use for this agent, you can also register system
                prompts via a function with [`system_prompt`][pydantic_ai.agent.Agent.system_prompt].
            deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
                parameterize the agent, and therefore get the best out of static type checking.
                If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
                or add a type hint `: Agent[None, <return type>]`.
            name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
                when the agent is first run.
            model_settings: Optional model request settings to use for this agent's runs, by default.
            retries: The default number of retries to allow for tool calls and output validation, before raising an error.
                For model request retries, see the [HTTP Request Retries](../retries.md) documentation.
            validation_context: Pydantic [validation context](https://docs.pydantic.dev/latest/concepts/validators/#validation-context) used to validate tool arguments and outputs.
            output_retries: The maximum number of retries to allow for output validation, defaults to `retries`.
            tools: Tools to register with the agent, you can also register tools via the decorators
                [`@agent.tool`][pydantic_ai.agent.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.agent.Agent.tool_plain].
            builtin_tools: The builtin tools that the agent will use. This depends on the model, as some models may not
                support certain tools. If the model doesn't support the builtin tools, an error will be raised.
            prepare_tools: Custom function to prepare the tool definition of all tools for each step, except output tools.
                This is useful if you want to customize the definition of multiple tools or you want to register
                a subset of tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
            prepare_output_tools: Custom function to prepare the tool definition of all output tools for each step.
                This is useful if you want to customize the definition of multiple output tools or you want to register
                a subset of output tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
            toolsets: Toolsets to register with the agent, including MCP servers and functions which take a run context
                and return a toolset. See [`ToolsetFunc`][pydantic_ai.toolsets.ToolsetFunc] for more information.
            defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
                it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
                which checks for the necessary environment variables. Set this to `false`
                to defer the evaluation until the first run. Useful if you want to
                [override the model][pydantic_ai.agent.Agent.override] for testing.
            end_strategy: Strategy for handling tool calls that are requested alongside a final result.
                See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
            instrument: Set to True to automatically instrument with OpenTelemetry,
                which will use Logfire if it's configured.
                Set to an instance of [`InstrumentationSettings`][pydantic_ai.agent.InstrumentationSettings] to customize.
                If this isn't set, then the last value set by
                [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all]
                will be used, which defaults to False.
                See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
            metadata: Optional metadata to store with each run.
                Provide a dictionary of primitives, or a callable returning one
                computed from the [`RunContext`][pydantic_ai.tools.RunContext] on each run.
                Metadata is resolved when a run starts and recomputed after a successful run finishes so it
                can reflect the final state.
                Resolved metadata can be read after the run completes via
                [`AgentRun.metadata`][pydantic_ai.agent.AgentRun],
                [`AgentRunResult.metadata`][pydantic_ai.agent.AgentRunResult], and
                [`StreamedRunResult.metadata`][pydantic_ai.result.StreamedRunResult],
                and is attached to the agent run span when instrumentation is enabled.
            history_processors: Optional list of callables to process the message history before sending it to the model.
                Each processor takes a list of messages and returns a modified list of messages.
                Processors can be sync or async and are applied in sequence.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools.
            tool_timeout: Default timeout in seconds for tool execution. If a tool takes longer than this,
                the tool is considered to have failed and a retry prompt is returned to the model (counting towards the retry limit).
                Individual tools can override this with their own timeout. Defaults to None (no timeout).
        """
        if agent is None:
            agent = pydantic_ai.Agent(
                model,
                output_type=output_type,
                instructions=instructions,
                system_prompt=system_prompt,
                deps_type=deps_type,
                name=name,
                model_settings=model_settings,
                retries=retries,
                validation_context=validation_context,
                output_retries=output_retries,
                tools=tools,
                builtin_tools=builtin_tools,
                prepare_tools=prepare_tools,
                prepare_output_tools=prepare_output_tools,
                toolsets=toolsets,
                defer_model_check=defer_model_check,
                end_strategy=end_strategy,
                instrument=instrument,
                metadata=metadata,
                history_processors=history_processors,
                event_stream_handler=event_stream_handler,
                tool_timeout=tool_timeout,
            )
        object.__setattr__(self, 'agent', agent)
        object.__setattr__(self, 'description', description)
        object.__setattr__(self, 'agent_calls', ensure_tuple(agent_calls))
        object.__setattr__(self, 'agent_handoffs', ensure_tuple(agent_handoffs))
        if name is None and agent.name is None:
            raise ValueError('Agent must have a name to participate in Collab')
        if memory is not None:
            if isinstance(memory, str):
                memory = [memory]
            if isinstance(memory, (list, tuple, set, frozenset)):
                memory = {ctx: 'rw' for ctx in memory}
            memory = {str_or_am_to_am(ctx): validate_r_rw(v) for ctx, v in memory.items()}
            if len(set(mem.name for mem in memory)) < len(memory):
                raise ValueError('Memory names must be unique.')
        object.__setattr__(self, 'memory', memory or {})
        object.__setattr__(self, 'name', name or agent.name)

    def __hash__(self) -> int:
        """Make CollabAgent hashable based on agent name."""
        return hash(hash(self.name) ^ hash(self.description))

    def __eq__(self, other: Any) -> bool:
        """Compare CollabAgent instances by agent name."""
        if isinstance(other, CollabAgent):
            return (
                self.name == other.name
                # self.agent == other.agent #and
                # self.description == other.description
            )
        return False

    def __repr__(self) -> str:
        """String representation showing name and description."""
        return f'CollabAgent(name={self.name!r}, description={self.description!r})'

    @property
    def requires_deps(self) -> bool:
        return self.agent.deps_type is not type(None)


# =============================================================================
# Output Types
# =============================================================================
class HandOffBase(BaseModel, Generic[T]):
    """Route to another agent for further processing.

    Use this when you finished with your work and want to hand off.
    If you just need help, call tools
    Example:
        HandoffOutput(
            next_agent="Expert",
            query="Please analyze this specific aspect..."
        )
    """

    reasoning: str | None = Field(
        default=None,
        description='Optional explanation of why this action was taken. Useful for debugging and transparency.',
    )
    next_agent: str = Field(description='Name of the agent to route to.')
    query: T = Field(description='The query/message/response to send to the next agent.')


# =============================================================================
# Context Data Structures
# =============================================================================


@dataclass
class PromptBuilderContext:
    """Context passed to prompt builder functions.

    Users can create custom prompt builders that accept this context.
    """

    agent: CollabAgent
    final_agent: bool
    can_handoff: bool
    handoff_agents: Sequence[CollabAgent]
    tool_agents: Sequence[CollabAgent]
    called_as_tool: bool
    ascii_topology: str | None = None
    can_do_parallel_agent_calls: bool = True
    context_info: dict[AgentMemory, list[str]] = field(default_factory=dict)


@dataclass(repr=False, kw_only=True)
class HandoffData:
    """Data passed during agent handoffs."""

    previous_handoff_str: str = ''
    caller_agent_name: str
    callee_agent_name: str
    message_history: list[Any] | None = None
    include_thinking: bool = False
    include_tool_calls_with_callee: bool = False


# =============================================================================
# Settings
# =============================================================================


@dataclass
class CollabSettings:
    """Network behavior configuration for agent Collabs.

    Controls how data flows between agents during handoffs and what information
    is included in context.
    Most options are `RegularOptions` Literals.
    disallow - Always no.
    allow - allow agent to choose.
    force - Always yes.
    """

    include_thinking: RegularOptions = 'disallow'
    """Control whether thinking parts are included in handoff context.
    "allow": Agent decides via HandoffOutput.include_thinking
    """

    include_conversation: RegularOptions = 'allow'
    """Control whether conversation history is included in handoff context.
    "allow": Agent decides via HandoffOutput.include_conversation
    """

    include_handoff: RegularOptions = 'allow'
    """Control whether previous handoff context is accumulated. 
    i.e. whether context from Agent A is sent by Agent B to Agent C when handing off.
    "allow": Agent decides via HandoffOutput.include_previous_handoff
    """

    include_tool_calls_with_callee: RegularOptions = 'allow'
    """Control whether tool calls and their results with target agents are included in handoff context.
    "allow": Agent decides via HandoffOutput.include_tool_calls_with_callee
    """

    output_restrictions: Literal['only_str', 'only_original', 'str_or_original'] = 'str_or_original'
    """Control what types agents can output in their query/response fields.
    - "only_str": Agents can only output strings
    - "only_original": Agents must use their configured _output_type
    - "str_or_original": Agents can output either string or their _output_type
    """

    prompt_builder: Callable[[PromptBuilderContext], str] | None = None
    """Custom function to build agent-specific instructions.
    If None, uses default_build_agent_prompt from _utils module.
    """

    context_builder: Callable[[HandoffData], str] | None = None
    """Custom function to build context string from handoff data.
    If None, uses get_context from _utils module.
    """

    include_topology_in_prompt: bool = True
    """If true, will include topology of the handoffs available between agents, 
    if handoffs are available"""


def generate_handoff_pydantic_model(settings: CollabSettings) -> type[HandOffBase[Any]]:
    """Dynamically create HandoffOutput class based on Collab settings.

    Converts "force"/"disallow" settings into ClassVar fields that enforce
    specific values rather than allowing agent control.

    Args:
        settings: The Collab network settings to apply

    Returns:
        A HandOffBase subclass with appropriate field types
    """
    static = ('force', 'disallow')

    class HandOffOutput(HandOffBase[T]):
        if settings.include_conversation in static:
            include_conversation: ClassVar[bool] = settings.include_conversation == 'force'
        else:
            include_conversation: bool = Field(
                default=False,
                description='Whether to include the full conversation history. '
                'Useful for complex multi-step reasoning.',
            )

        if settings.include_handoff in static:
            include_previous_handoff: ClassVar[bool] = settings.include_handoff == 'force'
        else:
            include_previous_handoff: bool = Field(
                default=False, description='Whether to include the previous handoff history.'
            )

        if settings.include_tool_calls_with_callee in static:
            include_tool_calls_with_callee: ClassVar[bool] = settings.include_tool_calls_with_callee == 'force'
        else:
            include_tool_calls_with_callee: bool = Field(
                default=False,
                description='Whether to include tool calls with that agent if you had any previous interactions.',
            )

        if settings.include_thinking in static:
            include_thinking: ClassVar[bool] = settings.include_thinking == 'force'
        else:
            include_thinking: bool = Field(
                default=False, description='Whether to include thinking parts from the conversation history.'
            )

    return HandOffOutput


# Used to redeclare it as is
# t_agent_desc already declared above.
