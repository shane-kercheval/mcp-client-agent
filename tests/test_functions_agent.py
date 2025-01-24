"""Tests for Function and Parameter classes."""
from dspy import Tool
from src.functions_agent import Function, Parameter, ParameterType

def test_parameter_basic():
    """Test basic Parameter initialization."""
    param = Parameter(
        name="test",
        type=ParameterType.STRING,
        required=True,
        description="A test parameter"
    )
    assert param.name == "test"
    assert param.type == ParameterType.STRING
    assert param.required is True
    assert param.description == "A test parameter"
    assert param.enum is None
    assert param.default is None
    assert param.any_of_schema is None

def test_parameter_with_enum():
    """Test Parameter with enum values."""
    param = Parameter(
        name="sort_order",
        type=ParameterType.STRING,
        required=True,
        description="Sort direction",
        enum=["asc", "desc"]
    )
    assert param.enum == ["asc", "desc"]

def test_parameter_with_default():
    """Test Parameter with default value."""
    param = Parameter(
        name="limit",
        type=ParameterType.INTEGER,
        required=False,
        description="Maximum items to return",
        default=10
    )
    assert param.default == 10

def test_parameter_with_any_of():
    """Test Parameter with anyOf schema."""
    any_of_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"}
        ],
        "default": None
    }
    param = Parameter(
        name="branch",
        type=ParameterType.ANY_OF,
        required=False,
        description="Branch name",
        any_of_schema=any_of_schema,
        default=None
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
                description="First argument"
            )
        ]
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
                        "description": "First argument"
                    }
                },
                "required": ["arg1"],
                "additionalProperties": False
            }
        }
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
                enum=["asc", "desc"]
            ),
            Parameter(
                name="limit",
                type=ParameterType.INTEGER,
                required=False,
                description="Maximum items",
                default=10
            ),
            Parameter(
                name="branch",
                type=ParameterType.ANY_OF,
                required=False,
                description="Branch name",
                any_of_schema={
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None
                }
            )
        ]
    )
    
    schema = function.to_openai_schema()
    assert schema["function"]["parameters"]["properties"]["sort_order"]["enum"] == ["asc", "desc"]
    assert schema["function"]["parameters"]["properties"]["limit"]["default"] == 10
    assert "anyOf" in schema["function"]["parameters"]["properties"]["branch"]

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
                description="First argument"
            )
        ]
    )
    
    tool = function.to_dspy_tool()
    assert isinstance(tool, Tool)
    assert tool.name == "test_function"
    assert tool.desc == "A test function"
    assert "arg1" in tool.args
    assert isinstance(tool.args["arg1"][0], type)  # type
    assert "First argument" in tool.args["arg1"][1]  # description

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
                enum=["asc", "desc"]
            )
        ]
    )
    
    tool = function.to_dspy_tool()
    assert "Allowed values: [asc, desc]" in tool.args["order"][1]

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
                default=10
            )
        ]
    )
    
    tool = function.to_dspy_tool()
    assert "Default value: 10" in tool.args["limit"][1]

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
            Parameter(name="any_of_arg", type=ParameterType.ANY_OF, required=True)
        ]
    )
    
    tool = function.to_dspy_tool()
    assert tool.args["str_arg"][0] == str  # noqa: E721
    assert tool.args["int_arg"][0] == int  # noqa: E721
    assert tool.args["float_arg"][0] == float  # noqa: E721
    assert tool.args["bool_arg"][0] == bool  # noqa: E721
    assert tool.args["array_arg"][0] == list  # noqa: E721
    assert tool.args["dict_arg"][0] == dict  # noqa: E721
    assert tool.args["any_of_arg"][0] == str  # noqa: E721
