"""Tests for MCP tool conversion functions."""
import pytest
from typing import Any
from src.utilities import convert_mcp_tools_to_functions, _convert_schema_to_parameter_type
from src.functions_agent import Function, Parameter, ParameterType

class MockMCPManager:
    """Mock MCP client manager for testing."""

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Mock tool execution."""
        return f"Called {tool_name} with {args}"

@pytest.fixture
def mock_manager():
    """Fixture providing mock manager."""
    return MockMCPManager()

@pytest.mark.parametrize(("schema_type", "expected_type"), [
    ("string", ParameterType.STRING),
    ("number", ParameterType.NUMBER),
    ("integer", ParameterType.INTEGER),
    ("boolean", ParameterType.BOOLEAN),
    ("object", ParameterType.DICT),
    ("array", ParameterType.ARRAY),
    ("unknown", ParameterType.STRING),  # Default case
])
def test_convert_schema_to_parameter_type(schema_type: str, expected_type: ParameterType):
    """Test schema type conversion."""
    schema = {"type": schema_type}
    result = _convert_schema_to_parameter_type(schema)
    assert result == expected_type

@pytest.mark.asyncio
async def test_convert_simple_tool(mock_manager: MockMCPManager):
    """Test converting a simple tool with one parameter."""
    tools = [{
        "name": "reverse_text",
        "description": "Reverse the input text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                },
            },
            "required": ["text"],
        },
    }]

    functions = await convert_mcp_tools_to_functions(tools, mock_manager)
    assert len(functions) == 1

    function = functions[0]
    assert isinstance(function, Function)
    assert function.name == "reverse_text"
    assert function.description == "Reverse the input text"

    assert len(function.parameters) == 1
    param = function.parameters[0]
    assert isinstance(param, Parameter)
    assert param.name == "text"
    assert param.type == ParameterType.STRING
    assert param.required is True
    result = await function.func(text="test")
    assert result == "Called reverse_text with {'text': 'test'}"

@pytest.mark.asyncio
async def test_convert_complex_tool(mock_manager: MockMCPManager):
    """Test converting a tool with multiple parameters of different types."""
    tools = [{
        "name": "analyze_data",
        "description": "Analyze numerical data",
        "inputSchema": {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                },
                "operation": {
                    "type": "string",
                },
                "precision": {
                    "type": "integer",
                },
            },
            "required": ["numbers", "operation"],
        },
    }]
    functions = await convert_mcp_tools_to_functions(tools, mock_manager)
    assert len(functions) == 1

    function = functions[0]
    assert isinstance(function, Function)
    assert function.name == "analyze_data"

    # Verify parameters
    assert len(function.parameters) == 3
    param_dict = {p.name: p for p in function.parameters}

    assert param_dict["numbers"].type == ParameterType.ARRAY
    assert param_dict["numbers"].required is True

    assert param_dict["operation"].type == ParameterType.STRING
    assert param_dict["operation"].required is True

    assert param_dict["precision"].type == ParameterType.INTEGER
    assert param_dict["precision"].required is False

    result = await function.func(numbers=[1, 2, 3], operation="sum", precision=2)
    assert result == "Called analyze_data with {'numbers': [1, 2, 3], 'operation': 'sum', 'precision': 2}"  # noqa: E501


@pytest.mark.asyncio
async def test_empty_tools(mock_manager: MockMCPManager):
    """Test handling empty tools list."""
    functions = await convert_mcp_tools_to_functions([], mock_manager)
    assert len(functions) == 0


@pytest.mark.asyncio
async def test_invalid_schema__raises_value_error(mock_manager: MockMCPManager):
    """Test handling invalid schema."""
    tools = [{
        "name": "bad_tool",
        "description": "Tool with invalid schema",
        "inputSchema": "invalid",  # Not a dict
    }]
    with pytest.raises(ValueError):  # noqa: PT011
        _ = await convert_mcp_tools_to_functions(tools, mock_manager)


@pytest.mark.asyncio
async def test_multiple_tools(mock_manager: MockMCPManager):
    """Test converting multiple tools."""
    tools = [
        {
            "name": "tool1",
            "description": "First tool",
            "inputSchema": {
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
                "required": ["arg1"],
            },
        },
        {
            "name": "tool2",
            "description": "Second tool",
            "inputSchema": {
                "type": "object",
                "properties": {"arg2": {"type": "number"}},
                "required": [],
            },
        },
    ]
    functions = await convert_mcp_tools_to_functions(tools, mock_manager)
    assert len(functions) == 2
    names = {f.name for f in functions}
    assert names == {"tool1", "tool2"}

    result = await functions[0].func(arg1="test1")
    assert result == "Called tool1 with {'arg1': 'test1'}"
    result = await functions[1].func(arg2=42)
    assert result == "Called tool2 with {'arg2': 42}"
