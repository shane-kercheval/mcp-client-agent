"""Tests for Function and Parameter classes."""
from dspy import Tool
from dotenv import load_dotenv
import os
import pytest
from src.functions_agent import (
    Function,
    FunctionAgent,
    FunctionCallResult,
    ModelConfiguration,
    Parameter,
    ParameterType,
    ToolChoiceType,
    ToolExecutionResultEvent,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def test_parameter_basic():
    """Test basic Parameter initialization."""
    param = Parameter(
        name="test",
        type=ParameterType.STRING,
        required=True,
        description="A test parameter",
    )
    assert param.name == "test"
    assert param.type == ParameterType.STRING
    assert param.required is True
    assert param.description == "A test parameter"
    assert param.enum is None
    assert param.default is None
    assert param.any_of_schema is None


def test_function_openai_no_parameters():
    """Test Function to OpenAI tool conversion without parameters."""
    function = Function(
        name="no_args",
        description="A function with no arguments",
    )
    tool = function.to_openai_schema()
    assert tool == {
        "type": "function",
        "function": {
            "name": "no_args",
            "description": "A function with no arguments",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def test_parameter_with_enum():
    """Test Parameter with enum values."""
    param = Parameter(
        name="sort_order",
        type=ParameterType.STRING,
        required=True,
        description="Sort direction",
        enum=["asc", "desc"],
    )
    assert param.enum == ["asc", "desc"]


def test_parameter_with_default():
    """Test Parameter with default value."""
    param = Parameter(
        name="limit",
        type=ParameterType.INTEGER,
        required=False,
        description="Maximum items to return",
        default=10,
    )
    assert param.default == 10


def test_parameter_with_any_of():
    """Test Parameter with anyOf schema."""
    any_of_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"},
        ],
        "default": None,
    }
    param = Parameter(
        name="branch",
        type=ParameterType.ANY_OF,
        required=False,
        description="Branch name",
        any_of_schema=any_of_schema,
        default=None,
    )
    assert param.any_of_schema == any_of_schema


def test_function_openai_schema_basic():
    """Test basic Function to OpenAI schema conversion."""
    function = Function(
        name="test_function",
        description="A test function",
        parameters=[
            Parameter(
                name="arg1",
                type=ParameterType.STRING,
                required=True,
                description="First argument",
            ),
        ],
    )

    schema = function.to_openai_schema()
    assert schema == {
        "type": "function",
        "function": {
            "name": "test_function",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {
                        "type": "string",
                        "description": "First argument",
                    },
                },
                "required": ["arg1"],
                "additionalProperties": False,
            },
        },
    }


def test_function_openai_schema_complex():
    """Test Function to OpenAI schema with all parameter features."""
    function = Function(
        name="complex_function",
        description="A complex function",
        parameters=[
            Parameter(
                name="sort_order",
                type=ParameterType.STRING,
                required=True,
                description="Sort direction",
                enum=["asc", "desc"],
            ),
            Parameter(
                name="limit",
                type=ParameterType.INTEGER,
                required=False,
                description="Maximum items",
                default=10,
            ),
            Parameter(
                name="branch",
                type=ParameterType.ANY_OF,
                required=False,
                description="Branch name",
                any_of_schema={
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            ),
        ],
    )
    schema = function.to_openai_schema()
    assert schema["function"]["parameters"]["properties"]["sort_order"]["enum"] == ["asc", "desc"]
    assert schema["function"]["parameters"]["properties"]["limit"]["default"] == 10
    assert "anyOf" in schema["function"]["parameters"]["properties"]["branch"]


def test_function_dspy_no_parameters():
    """Test Function to DSPy Tool conversion without parameters."""
    function = Function(
        name="no_args",
        description="A function with no arguments",
    )
    tool = function.to_dspy_tool()
    assert isinstance(tool, Tool)
    assert tool.name == "no_args"
    assert tool.desc == "A function with no arguments"
    assert not tool.args


def test_function_dspy_tool_basic():
    """Test basic Function to DSPy Tool conversion."""
    function = Function(
        name="test_function",
        description="A test function",
        parameters=[
            Parameter(
                name="arg1",
                type=ParameterType.STRING,
                required=True,
                description="First argument",
            ),
        ],
    )
    tool = function.to_dspy_tool()
    assert isinstance(tool, Tool)
    assert tool.name == "test_function"
    assert tool.desc == "A test function"
    assert "arg1" in tool.args
    assert tool.args["arg1"]["type"] == "string"
    assert tool.args["arg1"]["description"] == "First argument"


def test_function_dspy_tool_with_enum():
    """Test Function to DSPy Tool conversion with enum values."""
    function = Function(
        name="sort_function",
        description="A sorting function",
        parameters=[
            Parameter(
                name="order",
                type=ParameterType.STRING,
                required=True,
                description="Sort direction",
                enum=["asc", "desc"],
            ),
        ],
    )
    tool = function.to_dspy_tool()
    assert "Allowed values: [asc, desc]" in tool.args["order"]["description"]
    assert tool.args["order"]["enum"] == ["asc", "desc"]


def test_function_dspy_tool_with_default():
    """Test Function to DSPy Tool conversion with default values."""
    function = Function(
        name="paginate",
        description="A pagination function",
        parameters=[
            Parameter(
                name="limit",
                type=ParameterType.INTEGER,
                required=False,
                description="Page size",
                default=10,
            ),
        ],
    )
    tool = function.to_dspy_tool()
    assert "Default value: 10" in tool.args["limit"]["description"]


def test_function_dspy_tool_type_mapping():
    """Test type mapping in DSPy Tool conversion."""
    function = Function(
        name="type_test",
        description="Testing all types",
        parameters=[
            Parameter(name="str_arg", type=ParameterType.STRING, required=True),
            Parameter(name="int_arg", type=ParameterType.INTEGER, required=True),
            Parameter(name="float_arg", type=ParameterType.NUMBER, required=True),
            Parameter(name="bool_arg", type=ParameterType.BOOLEAN, required=True),
            Parameter(name="array_arg", type=ParameterType.ARRAY, required=True),
            Parameter(name="dict_arg", type=ParameterType.DICT, required=True),
            Parameter(name="any_of_arg", type=ParameterType.ANY_OF, required=True),
        ],
    )

    tool = function.to_dspy_tool()
    assert tool.args["str_arg"]["type"] == "string"
    assert tool.args["int_arg"]["type"] == "integer"
    assert tool.args["float_arg"]["type"] == "number"
    assert tool.args["bool_arg"]["type"] == "boolean"
    assert tool.args["array_arg"]["type"] == "array"
    assert tool.args["dict_arg"]["type"] == "object"
    assert tool.args["any_of_arg"]["type"] == "anyOf"


@pytest.fixture
def mock_calculator_function() -> Function:
    """Fixture providing a simple calculator function."""
    def multiply(numbers: list[int]) -> int:
        """Calculate the multiplication of numbers."""
        result = 1
        for num in numbers:
            result *= num
        return result

    return Function(
        name='calculate_multiply',
        description='Calculates the multiplication of numbers in the list provided',
        parameters=[
            Parameter(
                name='numbers',
                type=ParameterType.ARRAY,
                required=True,
                description='List of numbers to multiply',
            ),
        ],
        func=multiply,
    )


@pytest.fixture
def model_config() -> ModelConfiguration:
    """Fixture providing model configuration."""
    return ModelConfiguration(
        model='openai/gpt-4o-mini',
        api_key=OPENAI_API_KEY,
        temperature=0.0,
    )


@pytest.mark.asyncio
async def test_agent_single_tool_execution(
        mock_calculator_function: Function,
        model_config: ModelConfiguration,
    ) -> None:
    """Test agent executing a single tool successfully."""
    agent = FunctionAgent(
        model_config=model_config,
        tools=[mock_calculator_function],
        max_iters=3,
        choice_type=ToolChoiceType.REQUIRED,
    )
    result = await agent("What is the multiplication of 124, 194, and 315?")
    assert isinstance(result, FunctionCallResult)
    assert len(result.func_calls) == 1
    assert result.answer.replace(',', '') == str(124 * 194 * 315)
    assert result.reasoning
    assert result.token_usage.input_tokens > 0
    assert result.token_usage.output_tokens > 0
    assert result.token_usage.total_tokens == result.token_usage.input_tokens + result.token_usage.output_tokens  # noqa: E501
    assert result.token_usage.total_cost > 0

    tool_call = result.func_calls[0]
    assert tool_call.func_name == 'calculate_multiply'
    assert tool_call.func_args == {'numbers': [124, 194, 315]}
    assert tool_call.func_result == 124 * 194 * 315


@pytest.mark.asyncio
async def test_agent_no_tool_execution(
        mock_calculator_function: Function,
        model_config: ModelConfiguration,
    ) -> None:
    """Test agent handling a query that doesn't require tools."""
    agent = FunctionAgent(
        model_config=model_config,
        tools=[mock_calculator_function],  # incorrect tool
    )

    result = await agent('What is the capital of France?')
    assert isinstance(result, FunctionCallResult)
    assert len(result.func_calls) == 0
    assert 'Paris' in result.answer


@pytest.mark.asyncio
async def test_agent_tool_without_parameters(model_config: ModelConfiguration) -> None:
    """Test agent handling a tool without parameters."""
    def random_number_generator() -> str:
        return 42

    tool = Function(
        name='random_number_generator',
        description="Generates a random number.",
        func=random_number_generator,
    )
    agent = FunctionAgent(
        model_config=model_config,
        tools=[tool],
    )
    result = await agent('Generate a random number.')
    assert isinstance(result, FunctionCallResult)
    assert len(result.func_calls) == 1
    assert '42' in result.answer


@pytest.mark.asyncio
async def test_agent_event_callback(
        mock_calculator_function: Function,
        model_config: ModelConfiguration,
    ) -> None:
    """Test agent event callback system."""
    events = []
    agent = FunctionAgent(
        model_config=model_config,
        tools=[mock_calculator_function],
        callback=lambda event: events.append(event),
    )
    _ = await agent('What is the multiplication of 124, 194, and 315?')
    assert len(events) > 0
    tool_execution_result_event = next(
        (event for event in events if isinstance(event, ToolExecutionResultEvent)),
        None,
    )
    assert tool_execution_result_event
    assert tool_execution_result_event.tool_name == 'calculate_multiply'
    assert tool_execution_result_event.tool_args == {'numbers': [124, 194, 315]}
    assert tool_execution_result_event.result == 124 * 194 * 315
