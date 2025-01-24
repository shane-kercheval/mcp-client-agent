"""
Implements function calling for AI models with synchronous and asynchronous function execution
support. The current implementation subclasses DSPy's ReAct class to allow for prediction-only
responses (similar to OpenAI's Function Calling API) and structured output with detailed reasoning
steps and optional function calls.

This module provides a framework for defining, predicting, and executing tools/functions.
It supports both execution and prediction-only modes:

1. Execution Mode: When a Function instance has a callable `func` attribute, the system executes
   the function when the model predicts it.

2. Prediction Mode: When `func` is None, the system returns the model's function prediction without
   execution, matching OpenAI's Function Calling behavior.

Key Features:
- Type-safe function definitions with parameter validation
- Async function execution support
- Event-driven architecture for monitoring execution
- OpenAI-compatible function schemas

Example:
    >>> calculator = Function(
    ...     name="add",
    ...     parameters=[
    ...         Parameter("x", ParameterType.NUMBER, required=True),
    ...         Parameter("y", ParameterType.NUMBER, required=True)
    ...     ],
    ...     func=lambda x, y: x + y
    ... )
    )
    >>> agent = FunctionAgent(model_config, tools=[calculator])
    >>> result = await agent("What is 2 + 2?")
"""
import asyncio
from dataclasses import dataclass, asdict, field
from enum import Enum
import time
from typing import Callable
import dspy


@dataclass
class Message:
    """Represents a message in a conversation."""

    role: str
    content: str

    def to_dict(self) -> dict:
        """Convert to a dictionary."""
        return asdict(self)

class ParameterType(Enum):
    """
    Enumeration of OpenAI parameter types.

    NOTE: "Object" in the specificaition is meant to represent a JSON object (or dictionary) types,
    so the enum name is changed to "DICT" to avoid confusion; the "object" value is used when
    converting to OpenAI schema format, and `dict` is used when generating DSPy Tools.

    https://platform.openai.com/docs/guides/structured-outputs#supported-types
    """

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    DICT = "object"
    ARRAY = "array"
    ENUM = "enum"
    ANY_OF = "anyOf"

@dataclass
class Parameter:
    """
    Represents a parameter of the Function.

    For example,

        Parameter(
            name='temperature_units',
            type=ParameterType.STRING,
            required=True,
            description='The units of the temperature value',
            enum=['Celsius', 'Fahrenheit', 'Kelvin']
        )

    Attributes:
        name: The parameter name.
        type: The parameter type.
        required: Whether the parameter is required.
        description: Description of the parameter.
        enum: List of possible values for the parameter.
        default: Default value for the parameter.
        any_of_schema: List of schemas for the parameter, if present.
    """

    name: str
    type: ParameterType
    required: bool
    description: str | None = None
    enum: list[str] | None = None
    default: object | None = None
    any_of_schema: list[dict] | None = None

@dataclass
class Function:
    """Represents a function that can be called by the model."""

    name: str
    parameters: list[Parameter]
    description: str | None = None
    func: Callable | None = None

    def to_openai_schema(self) -> dict[str, object]:
        """Convert to OpenAI function format."""
        properties = {}
        required = []

        for param in self.parameters:
            if param.any_of_schema:  # noqa: SIM108
                # Use the original anyOf schema
                param_dict = param.any_of_schema
            else:
                param_dict = {"type": param.type.value}

            if param.description:
                param_dict["description"] = param.description
            if param.enum:
                param_dict["enum"] = param.enum
            if param.default is not None:
                param_dict["default"] = param.default

            properties[param.name] = param_dict
            if param.required:
                required.append(param.name)

        parameters_dict = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters_dict["required"] = required
        parameters_dict["additionalProperties"] = False

        return {
            "type": "function",
            "function": {
                "name": self.name,
                **({"description": self.description} if self.description else {}),
                "parameters": parameters_dict,
            },
        }

    def to_dspy_tool(self) -> dspy.Tool:
        """Convert to DSPy Tool."""
        type_mapping = {
            ParameterType.STRING: str,
            ParameterType.NUMBER: float,
            ParameterType.INTEGER: int,
            ParameterType.BOOLEAN: bool,
            ParameterType.ARRAY: list,
            ParameterType.DICT: dict,
            ParameterType.ANY_OF: str,
            ParameterType.ENUM: str,
        }
        args = {}
        for param in self.parameters:
            description = param.description or ""
            if param.enum:
                if description:
                    description += "\n"
                description += f"Allowed values: [{', '.join(param.enum)}]"
            if param.default is not None:
                if description:
                    description += "\n"
                description += f"Default value: {param.default}"
            if param.any_of_schema:
                if description:
                    description += "\n"
                description += f"Accepts: {param.any_of_schema}"

            args[param.name] = (
                type_mapping.get(param.type, str),
                description,
            )
        if self.func is None:
            ###################################################################################
            # Monkey-patch the dspy.Tool.__init__ method to allow for creating tools without a
            # function
            ###################################################################################
            original_tool_init = dspy.Tool.__init__
            def _patched_tool_init(self, func: Callable | None = None, name: str = None, desc: str = None, args: dict[str, object] = None):  # noqa
                if not name or not desc or not args:
                    raise ValueError("When func is None, name, desc, and args must be provided")
                self.func = None
                self.name = name
                self.desc = desc
                self.args = args
            # Apply the patch
            dspy.Tool.__init__ = _patched_tool_init
            tool = dspy.Tool(
                func=None,
                name=self.name,
                desc=self.description,
                args=args,
            )
            # Revert the patch
            dspy.Tool.__init__ = original_tool_init
            ###################################################################################
            # end of monkey-patch
            ###################################################################################
        else:
            tool = dspy.Tool(
                func=self.func,
                name=self.name,
                desc=self.description,
                args=args,
            )
        return tool


@dataclass
class FunctionCall:
    """
    Represents a function call/prediction.

    If the Function `func` attribute has a callable function,
    the `func_result` will contain the result of the function call. Otherwise, the `func_result`
    will be None.

    The `thought` attribute contains the reasoning step that led to the function call.
    """

    func_name: str
    func_args: dict[str, object]
    func_result: str | None
    thought: str


@dataclass
class FunctionCallResult:
    """Represents the result of a ReAct prediction with tool calls."""

    answer: str
    reasoning: str
    func_calls: list[FunctionCall]


class ToolChoiceType(Enum):
    """
    The intent is to implement similar functionality as OpenAI Function Calling.
    You can force specific behavior with the tool_choice parameter.

    Auto: (Default) Call zero, one, or multiple functions.
    Required: Call one or more functions.
    Forced Function: Call exactly one specific function.
    """

    AUTO = "auto"
    REQUIRED = "required"

@dataclass
class ModelConfiguration:
    """Configuration for the model and tools."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 10000


@dataclass
class AgentEvent:
    """Base class for all agent events."""

    iteration: int

@dataclass
class ThinkStartEvent(AgentEvent):
    """Agent is starting to think/reason."""

    inputs: dict[str, object]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThoughtEvent(AgentEvent):
    """Agent produced a thought and tool prediction."""

    thought: str
    tool_name: str | None
    tool_args: dict[str, object] | None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolExecutionStartEvent(AgentEvent):
    """Agent is starting to execute a tool."""

    tool_name: str
    tool_args: dict[str, object]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolExecutionResultEvent(AgentEvent):
    """Tool execution result."""

    tool_name: str
    tool_args: dict[str, object]
    result: object
    timestamp: float = field(default_factory=time.time)


class FunctionAgent(dspy.ReAct):
    """
    An extension of DSPy's ReAct that provides allows similar functionality to OpenAI's Function
    Calling API.


    The FunctionAgent class enhances DSPy's ReAct by providing:
        1. Prediction-only mode - Returns function predictions without execution when `func=None`
        2. Async/sync support - Native support for both synchronous and asynchronous tool functions
        3. Structured output - Returns detailed reasoning steps and function calls in
           FunctionCallResult format
        4. Event system - Emits events for monitoring the reasoning process
        5. Type-safe interfaces - Strong typing for tools, parameters, and message passing

    This class allows the user to define a list of tools as Function instances, which can
    optionally include a callable function to execute the tool when predicted by the model. If
    the function is not provided, the tool call prediction will be returned without executing the
    tool function.

    The class returns results in a structured format using FunctionCallResult, which contains both
    the final prediction and a detailed log of all tool interactions.

    Attributes:
        Inherits all attributes from dspy.ReAct

    Example:
        >>> # Define a search function
        >>> search = Function(
        ...     name="search",
        ...     description="Search for information on a topic",
        ...     parameters=[
        ...         Parameter("query", ParameterType.STRING, required=True),
        ...         Parameter("max_results", ParameterType.INTEGER, required=False)
        ...     ],
        ...     func=async_search_function
        ... )
        >>> # Initialize agent
        >>> agent = FunctionAgent(
        ...     model_config=ModelConfiguration(model="gpt-4"),
        ...     tools=[search],
        ...     max_iters=3
        ... )
        >>> # Run inference
        >>> result = await agent("Who wrote The Brothers Karamazov?")
        >>> print(f"Answer: {result.answer}")
        >>> print(f"Tool calls: {len(result.func_calls)}")
            ```
    """

    def __init__(
            self,
            model_config: ModelConfiguration,
            tools: list[Function],
            max_iters: int = 5,
            choice_type: ToolChoiceType = ToolChoiceType.AUTO,
            callback: Callable[[AgentEvent], None] | None = None,
        ):
        """
        Initializes a FunctionAgent for AI-driven function calling and reasoning.

        Args:
            model_config:
                Configuration settings for the language model including:
            tools:
                Available tools/functions for the agent to use. Two modes supported:
                1. Prediction-only mode (func=None):
                    Returns model's function predictions without execution
                2. Execution mode (func=callable):
                    Executes predicted function calls automatically
            max_iters:
                Maximum reasoning steps before forcing completion.
                Default: 5
            choice_type:
                Controls function calling behavior:
                - AUTO (default): Model freely chooses to use 0+ functions
                - REQUIRED: Model must use at least one function
                Default: ToolChoiceType.AUTO
            callback:
                Event handler for monitoring agent execution. Receives events for:
                - Thought generation
                - Tool selection
                - Tool execution
                - Results
        """
        self._callback = callback
        lm = dspy.LM(
            model=model_config.model,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            cache=False,
        )
        dspy.configure(lm=lm)

        if choice_type == ToolChoiceType.AUTO:
            base_instructions = "Answer questions by using the available tools when helpful."
        elif choice_type == ToolChoiceType.REQUIRED:
            base_instructions = "You must use at least one tool to answer the question."
        else:
            raise ValueError(f"Unsupported ToolChoiceType: {choice_type}")

        base_instructions += " If there is an error in the tool, either try to resolve the error if it is fixable, or use the 'finish' tool to end the reasoning process and report the error."  # noqa: E501
        base_signature = dspy.Signature("question -> answer", instructions=base_instructions)
        super().__init__(
            base_signature,
            tools=[tool.to_dspy_tool() for tool in tools],
            max_iters=max_iters,
        )

    async def __call__(self, messages: str | list[Message]) -> FunctionCallResult:
        """
        Overrides the __call__ method in dspy.ReAct to allow for passing a string or list of
        Message instances to the model.
        """
        if isinstance(messages, str):
            question = messages
        elif isinstance(messages, list):
            for i, message in enumerate(messages):
                if isinstance(message, Message):
                    messages[i] = message.to_dict()
            if len(messages) == 1:
                question = messages[0]['content']
            else:
                # otherwise we need to build up a string with ROLE: content\nROLE: content\n...
                question = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        else:
            raise ValueError("messages must be a string or a list of Message instances or dicts")
        return await self.forward(question=question)

    async def _emit(self, event: AgentEvent) -> None:
        if self._callback:
            if asyncio.iscoroutinefunction(self._callback):
                await self._callback(event)
            else:
                self._callback(event)
                # await asyncio.get_event_loop().run_in_executor(
                #     None,
                #     self._callback,
                #     event,
                # )


    async def forward(self, **input_args: dict[str, object]) -> FunctionCallResult:
        """
        Overrides the forward method in dspy.ReAct to capture and return each step of the
        reasoning process, including thoughts, tool calls, and their results. It maintains the same
        core ReAct functionality while providing more visibility into the execution process.
        """
        def _format(trajectory: dict, last_iteration: bool) -> str:  # noqa: ARG001
            adapter = dspy.settings.adapter or dspy.ChatAdapter()
            trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
            return adapter.format_fields(trajectory_signature, trajectory, role="user")

        trajectory = {}
        tool_predictions = []
        for index in range(self.max_iters):
            await self._emit(ThinkStartEvent(iteration=index, inputs=input_args))
            # pred = self.react(
            #     **input_args,
            #     trajectory=_format(trajectory,
            #     last_iteration=(idx == self.max_iters - 1)),
            # )

            # execute the react function (which calls llm) using async I/O
            pred = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.react(
                    **input_args,
                    trajectory=_format(trajectory, last_iteration=(index == self.max_iters - 1)),
                ),
            )
            has_tool_prediction = pred.next_tool_name != "finish" and pred.next_tool_name
            await self._emit(ThoughtEvent(
                iteration=index,
                thought=pred.next_thought,
                tool_name=pred.next_tool_name if has_tool_prediction else None,
                tool_args=pred.next_tool_args,
            ))
            trajectory[f"thought_{index}"] = pred.next_thought
            trajectory[f"tool_name_{index}"] = pred.next_tool_name
            trajectory[f"tool_args_{index}"] = pred.next_tool_args
            if not has_tool_prediction:
                break
            try:
                tool = self.tools[pred.next_tool_name]
                if tool.func is None:
                    tool_predictions.append(FunctionCall(
                        thought=pred.next_thought,
                        func_name=pred.next_tool_name,
                        func_args=pred.next_tool_args,
                        func_result=None,
                    ))
                    return FunctionCallResult(
                        answer=None,
                        reasoning=None,
                        func_calls=tool_predictions,
                    )
                await self._emit(ToolExecutionStartEvent(
                    iteration=index,
                    tool_name=pred.next_tool_name,
                    tool_args=pred.next_tool_args,
                ))
                # If the tool.func is an async function, await it
                if asyncio.iscoroutinefunction(tool.func):
                    result = await tool(**pred.next_tool_args)
                else:
                    result = tool(**pred.next_tool_args)
                    # result = await asyncio.get_event_loop().run_in_executor(
                    #     None, lambda: tool(**pred.next_tool_args)
                    # )
                await self._emit(ToolExecutionResultEvent(
                    iteration=index,
                    tool_name=pred.next_tool_name,
                    tool_args=pred.next_tool_args,
                    result=result,
                ))
                trajectory[f"observation_{index}"] = result
                tool_predictions.append(FunctionCall(
                    thought=pred.next_thought,
                    func_name=pred.next_tool_name,
                    func_args=pred.next_tool_args,
                    func_result=result,
                ))
            except Exception as e:
                error_msg = f"Failed to execute: {e}"
                trajectory[f"observation_{index}"] = error_msg
                tool_predictions.append(FunctionCall(
                    thought=pred.next_thought,
                    func_name=pred.next_tool_name,
                    func_args=pred.next_tool_args,
                    func_result=error_msg,
                ))
        extract = self.extract(**input_args, trajectory=_format(trajectory, last_iteration=False))
        return FunctionCallResult(
            answer=extract.answer,
            reasoning=extract.reasoning,
            func_calls=tool_predictions,
        )
