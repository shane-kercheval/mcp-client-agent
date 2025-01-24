"""Tests for MCP tool conversion functions."""
import pytest
from typing import Any
from src.utilities import convert_mcp_tools_to_functions, _convert_schema_to_parameter
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


def test_convert_schema_to_parameter_basic():
    """Test basic schema conversion."""
    schema = {
        "type": "string",
        "description": "A test parameter",
        "title": "Test"
    }
    param = _convert_schema_to_parameter("test_param", schema, required=True)
    
    assert isinstance(param, Parameter)
    assert param.name == "test_param"
    assert param.type == ParameterType.STRING
    assert param.required is True
    assert param.description == "A test parameter"
    assert param.default is None
    assert param.any_of_schema is None


def test_convert_schema_to_parameter_with_default():
    """Test schema conversion with default value."""
    schema = {
        "type": "integer",
        "title": "Max Count",
        "default": 10
    }
    param = _convert_schema_to_parameter("max_count", schema, required=False)
    
    assert param.type == ParameterType.INTEGER
    assert param.default == 10
    assert param.description == "Max Count"


def test_convert_schema_to_parameter_anyof():
    """Test schema conversion with anyOf."""
    schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"}
        ],
        "default": None,
        "title": "Base Branch"
    }
    param = _convert_schema_to_parameter("base_branch", schema, required=False)
    
    assert param.type == ParameterType.ANY_OF
    assert param.default is None
    assert param.any_of_schema == schema
    assert param.description == "Base Branch"


def test_convert_schema_to_parameter_anyof_with_nested_default():
    """Test schema conversion with default inside anyOf options."""
    schema = {
        "anyOf": [
            {"type": "string", "default": "main"},
            {"type": "null"}
        ],
        "title": "Base Branch"
    }
    param = _convert_schema_to_parameter("base_branch", schema, required=False)
    
    assert param.type == ParameterType.ANY_OF
    assert param.default == "main"


def test_convert_schema_to_parameter_with_enum():
    """Test schema conversion with enum values."""
    schema = {
        "type": "string",
        "enum": ["asc", "desc"],
        "description": "Sort direction"
    }
    param = _convert_schema_to_parameter("sort_order", schema, required=True)
    
    assert param.type == ParameterType.STRING
    assert param.enum == ["asc", "desc"]
    assert param.description == "Sort direction"


def test_convert_schema_to_parameter_array():
    """Test schema conversion for array type."""
    schema = {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of file paths"
    }
    param = _convert_schema_to_parameter("files", schema, required=True)
    
    assert param.type == ParameterType.ARRAY
    assert param.description == "List of file paths"


def test_convert_schema_to_parameter_complex_object():
    """Test schema conversion for complex object type."""
    schema = {
        "type": "object",
        "properties": {
            "nested": {"type": "string"}
        },
        "description": "A complex object"
    }
    param = _convert_schema_to_parameter("complex", schema, required=True)
    
    assert param.type == ParameterType.DICT
    assert param.description == "A complex object"


@pytest.mark.asyncio
async def test_convert_mcp_tools_git_create_branch(mock_manager):
    """Test converting git create branch tool."""
    tools = [{
        "server": "git",
        "name": "git_create_branch",
        "description": "Creates a new branch from an optional base branch",
        "inputSchema": {
            "properties": {
                "repo_path": {
                    "title": "Repo Path",
                    "type": "string"
                },
                "branch_name": {
                    "title": "Branch Name",
                    "type": "string"
                },
                "base_branch": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "title": "Base Branch"
                }
            },
            "required": ["repo_path", "branch_name"],
            "title": "GitCreateBranch",
            "type": "object"
        }
    }]
    
    functions = await convert_mcp_tools_to_functions(tools, mock_manager)
    assert len(functions) == 1
    
    function = functions[0]
    params = {p.name: p for p in function.parameters}
    
    assert params["repo_path"].type == ParameterType.STRING
    assert params["repo_path"].required is True
    
    assert params["branch_name"].type == ParameterType.STRING
    assert params["branch_name"].required is True
    
    assert params["base_branch"].type == ParameterType.ANY_OF
    assert params["base_branch"].required is False
    assert params["base_branch"].default is None


@pytest.mark.asyncio
async def test_convert_mcp_tools_git_log(mock_manager):
    """Test converting git log tool with default value."""
    tools = [{
        "server": "git",
        "name": "git_log",
        "description": "Shows the commit logs",
        "inputSchema": {
            "properties": {
                "repo_path": {
                    "title": "Repo Path",
                    "type": "string"
                },
                "max_count": {
                    "default": 10,
                    "title": "Max Count",
                    "type": "integer"
                }
            },
            "required": ["repo_path"],
            "title": "GitLog",
            "type": "object"
        }
    }]
    
    functions = await convert_mcp_tools_to_functions(tools, mock_manager)
    assert len(functions) == 1
    
    function = functions[0]
    params = {p.name: p for p in function.parameters}
    
    assert params["max_count"].type == ParameterType.INTEGER
    assert params["max_count"].required is False
    assert params["max_count"].default == 10
    assert params["max_count"].description == "Max Count"