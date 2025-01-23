"""Core MCP client functionality."""
from dataclasses import dataclass
import json
from contextlib import AsyncExitStack
from textwrap import dedent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None

    def get_params(self) -> StdioServerParameters:
        """Convert config to StdioServerParameters."""
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )


@dataclass
class ToolInfo:
    """Information about a tool."""

    server_name: str
    description: str | None
    input_schema: dict


class MCPClientManager:
    """Manages connections to MCP servers."""

    def __init__(self):
        self.servers: dict[str, ClientSession] = {}
        self.tool_map: dict[str, ToolInfo] = {}
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        await self.cleanup()

    @staticmethod
    def load_config(config_path: str | Path) -> list[ServerConfig]:
        """
        Load server configurations from config file (same format as Claude Desktop).

        Example config file:

        ```
        {
            "mcpServers": {
                "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/Users/username/Desktop",
                    "/Users/username/Downloads"
                ]
                }
            }
        }
        ```
        """
        with open(config_path) as f:
            config = json.load(f)
        servers = []
        for name, server_config in config.get("mcpServers", {}).items():
            servers.append(ServerConfig(
                name=name,
                command=server_config["command"],
                args=server_config["args"],
                env=server_config.get("env"),
            ))
        return servers

    async def connect_servers(self, config_path: Path) -> None:
        """
        Connect to all servers in the config file.

        Example config file:

        ```
        {
            "mcpServers": {
                "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/Users/username/Desktop",
                    "/Users/username/Downloads"
                ]
                }
            }
        }
        ```
        """
        configs = MCPClientManager.load_config(config_path)
        for config in configs:
            try:
                await self.connect_server(config)
                print(f"Connected to {config.name}")
            except Exception as e:
                print(f"Error connecting to {config.name}: {e}")

    async def connect_server(self, config: ServerConfig) -> None:
        """
        Connect to an MCP server and register its tools.

        Raises:
            ValueError: If a tool name conflicts with an existing tool from another server.
        """
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(config.get_params()),
        )
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio_transport[0], stdio_transport[1]),
        )
        await session.initialize()
        self.servers[config.name] = session

        # Get and register tools
        response = await session.list_tools()
        for tool in response.tools:
            if tool.name in self.tool_map:
                existing = self.tool_map[tool.name]
                raise ValueError(
                    f"Tool name conflict: '{tool.name}' is provided by both "
                    f"'{existing.server_name}' and '{config.name}'",
                )
            self.tool_map[tool.name] = ToolInfo(
                server_name=config.name.strip(),
                description=dedent(tool.description).strip(),
                input_schema=tool.inputSchema,
            )

    async def list_tools(self) -> list[dict]:
        """List all available tools across all servers."""
        return [{
            "server": info.server_name,
            # below key/values are consistent with the MCP API
            "name": name,
            "description": info.description,
            "inputSchema": info.input_schema,
        } for name, info in self.tool_map.items()]

    async def call_tool(self, tool_name: str, args: dict[str, object]) -> str:
        """
        Call a tool by name, automatically selecting the appropriate server.

        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool

        Raises:
            ValueError: If the tool name is not found
        """
        tool_info = self.tool_map.get(tool_name)
        if not tool_info:
            available_tools = ", ".join(f"'{t}'" for t in self.tool_map)
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {available_tools}",
            )

        result = await self.servers[tool_info.server_name].call_tool(tool_name, args)
        return result.content

    async def cleanup(self) -> None:
        """Clean up all connections."""
        await self.exit_stack.aclose()
