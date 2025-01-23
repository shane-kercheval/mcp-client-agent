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


def _convert_schema_to_parameter_type(schema: dict[str, object]) -> ParameterType:
    """Convert JSON schema type to ParameterType, handling array item types."""
    type_str = schema["type"]
    if type_str == "array" and "items" in schema:
        # Could potentially use the items type info for better type hints
        # e.g., array of numbers vs array of strings
        return ParameterType.ARRAY
    return {
        "string": ParameterType.STRING,
        "number": ParameterType.NUMBER,
        "integer": ParameterType.INTEGER,
        "boolean": ParameterType.BOOLEAN,
        "object": ParameterType.DICT,
        "array": ParameterType.ARRAY,
    }.get(type_str, ParameterType.STRING)


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
        parameters = []
        schema = tool["inputSchema"]  # Note: using inputSchema instead of schema
        if not isinstance(schema, dict):
            raise ValueError(f"Invalid schema for tool {tool['name']}")
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for name, prop in properties.items():
            param_type = _convert_schema_to_parameter_type(prop)
            parameters.append(Parameter(
                name=name,
                type=param_type,
                required=name in required,
                description=None,
            ))
        functions.append(Function(
            name=tool["name"],
            description=tool["description"],
            parameters=parameters,
            func=await make_wrapper(tool["name"]),
        ))

    return functions
