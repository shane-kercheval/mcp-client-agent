"""Utility functions."""
import re
from src.functions_agent import Function, Parameter, ParameterType
from src.mcp_manager import MCPClientManager


def colorize_markdown(text: str) -> str:
    """Colorizes text (used as the output in the terminal) based on markdown syntax."""
    # Check for code blocks surrounded by ```
    if re.search(r'```.*?```', text, flags=re.DOTALL):
        # Apply an approximate orange color and bold formatting for code blocks
        # 38;5;208 is an ANSI escape code for a color close to orange
        text = re.sub(r'```(.*?)```', r'\033[38;5;208;1m\1\033[0m', text, flags=re.DOTALL)
    # Apply blue color and bold for text surrounded by **
    text = re.sub(r'\*\*(.*?)\*\*', r'\033[34;1m\1\033[0m', text)
    # Apply green color and bold for text surrounded by ==
    text = re.sub(r'==(.*?)==', r'\033[32;1m\1\033[0m', text)
    # Apply orange color and bold for text surrounded by `
    text = re.sub(r'`(.*?)`', r'\033[38;5;208;1m\1\033[0m', text)
    # text = re.sub(r'`(.*?)`', r'\033[38;5;208m\1\033[0m', text)
    return text  # noqa: RET504


def colorize_gray(text: str) -> str:
    """Colorizes text (used as the output in the terminal) to gray."""
    # Apply gray color to all text
    return f'\033[90m{text}\033[0m'


def colorize_green(text: str) -> str:
    """Colorizes text (used as the output in the terminal) to green."""
    # Apply green color to all text
    return f'\033[32m{text}\033[0m'


def colorize_orange(text: str) -> str:
    """Colorizes text (used as the output in the terminal) to orange."""
    # Apply orange color to all text
    return f'\033[38;5;208m{text}\033[0m'


def colorize_red(text: str) -> str:
    """Colorizes text (used as the output in the terminal) to red."""
    # Apply red color to all text
    return f'\033[31m{text}\033[0m'


def colorize_blue(text: str) -> str:
    """Colorizes text (used as the output in the terminal) to blue."""
    # Apply blue color to all text
    return f'\033[34m{text}\033[0m'


def _convert_schema_to_parameter(name: str, schema: dict[str, object], required: bool) -> Parameter:  # noqa: E501
    """
    Convert a JSON schema parameter definition into a Parameter object.

    Examples:
        Basic string parameter:
        >>> schema = {"type": "string", "description": "The user's name"}
        >>> param = _convert_schema_to_parameter("username", schema, required=True)

        Integer with default:
        >>> schema = {
        ...     "type": "integer",
        ...     "title": "Max Count",
        ...     "default": 10
        ... }
        >>> param = _convert_schema_to_parameter("max_count", schema, required=False)

        Parameter with anyOf (union type):
        >>> schema = {
        ...     "anyOf": [
        ...         {"type": "string"},
        ...         {"type": "null"}
        ...     ],
        ...     "default": None,
        ...     "title": "Base Branch"
        ... }
        >>> param = _convert_schema_to_parameter("base_branch", schema, required=False)

        Enum parameter:
        >>> schema = {
        ...     "type": "string",
        ...     "enum": ["asc", "desc"],
        ...     "description": "Sort direction"
        ... }
        >>> param = _convert_schema_to_parameter("sort_order", schema, required=True)

        Array parameter:
        >>> schema = {
        ...     "type": "array",
        ...     "items": {"type": "string"},
        ...     "description": "List of file paths"
        ... }
        >>> param = _convert_schema_to_parameter("files", schema, required=True)

    Notes:
        - For `anyOf` schemas, the parameter type is set to ANY_OF and the original schema is
          preserved
        - When no type is specified, defaults to STRING type with a warning
        - Title is used as description if no description is provided

    Args:
        name: Name of the parameter
        schema: JSON schema definition for the parameter
        required: Whether this parameter is required
    """
    default_value = schema.get('default')
    description = schema.get('description') or schema.get('title')

    if 'anyOf' in schema:
        if default_value is None:
            # Look for default in anyOf options
            for option in schema['anyOf']:
                if isinstance(option, dict) and 'default' in option:
                    default_value = option['default']
                    break
        return Parameter(
            name=name,
            type=ParameterType.ANY_OF,
            required=required,
            description=description,
            default=default_value,
            any_of_schema=schema,
        )

    if 'type' not in schema:
        raise ValueError(f"Missing 'type' in schema for parameter `{name}`: `{schema}`")

    type_str = schema['type']
    type_mapping = {
        "string": ParameterType.STRING,
        "number": ParameterType.NUMBER,
        "integer": ParameterType.INTEGER,
        "boolean": ParameterType.BOOLEAN,
        "object": ParameterType.DICT,
        "array": ParameterType.ARRAY,
        "enum": ParameterType.ENUM,
    }
    return Parameter(
        name=name,
        type=type_mapping.get(type_str, ParameterType.STRING),
        required=required,
        description=description,
        default=default_value,
        enum=schema.get('enum'),
    )

async def convert_mcp_tools_to_functions(
        tools: list[dict],
        manager: MCPClientManager,
    ) -> list[Function]:
    """
    Convert MCP server tools into Function objects that can be used by FunctionAgent.

    According to this documentation, https://modelcontextprotocol.io/docs/concepts/tools
    MCP tools have the following structure:

    ```
    {
        name: string;          // Unique identifier for the tool
        description?: string;  // Human-readable description
        inputSchema: {         // JSON Schema for the tool's parameters
            type: "object",
            properties: { ... }  // Tool-specific parameters
        }
    }
    ```

    Args:
        tools: List of tool definitions from MCP server containing name, description, and parameter
        schema
        manager: MCPClientManager instance used to execute the tools

    Returns:
        List of Function objects, each wrapping an MCP tool with an async execution function

    Example:
        tools = [
            {
                "name": "search",
                "description": "Search for documents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"}
                    },
                    "required": ["query"]
                }
            }
        ]
        functions = await convert_mcp_tools_to_functions(tools, manager)
    """
    functions = []
    for tool in tools:
        async def make_wrapper(tool_name: str):  # noqa: ANN202
            async def wrapper(**kwargs: object) -> str:
                return await manager.call_tool(tool_name, kwargs)
            return wrapper

        schema = tool.get('inputSchema')
        if not schema:
            raise ValueError(f"Missing 'inputSchema' for tool {tool['name']}: {tool}")
        if not isinstance(schema, dict):
            raise ValueError(f"Invalid schema for tool {tool['name']}: {schema}")

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        parameters = [
            _convert_schema_to_parameter(
                name=name,
                schema=prop,
                required=name in required,
            )
            for name, prop in properties.items()
        ]

        functions.append(Function(
            name=tool["name"],
            description=tool["description"],
            parameters=parameters,
            func=await make_wrapper(tool["name"]),
        ))

    return functions
