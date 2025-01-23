"""CLI interface for MCP client."""
import asyncio
import json
import os
from pathlib import Path
import click
from src.mcp_manager import MCPClientManager
from src.functions_agent import (
    FunctionAgent,
    ModelConfiguration,
    Message,
    AgentEvent,
    ThinkStartEvent,
    ThoughtEvent,
    ToolExecutionResultEvent,
    ToolExecutionStartEvent,
)
from src.utilities import (
    colorize_gray,
    colorize_markdown,
    colorize_red,
    convert_mcp_tools_to_functions,
)

DEFAULT_CONFIG_PATHS = [
    Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
    Path.home() / "mcp_config.json",
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@click.group()
@click.option(
    '--config',
    type=click.Path(exists=True),
    help='Path to Claude Desktop config file. If not provided, will look in default locations.',
)
@click.option(
    '--model',
    type=str,
    help='Model to use for the FunctionAgent.',
)
@click.option(
    '--base_url',
    type=str,
    help='base_url of OpenAI compatible server.',
)
@click.pass_context
def cli(ctx: click.Context, config: str | None, model: str | None, base_url: str | None) -> None:
    """MCP client CLI."""
    if config:
        config_path = Path(config)
    else:
        # Try default locations
        for path in DEFAULT_CONFIG_PATHS:
            if path.exists():
                config_path = path
                break
        else:
            raise click.ClickException(
                "No config file found. Please provide --config or create a config file in default locations.",  # noqa: E501
            )
    if base_url:
        model = 'openai/test'

    ctx.obj = {
        'config_path': config_path,
        'model': model,
        'base_url': base_url,
    }

async def _chat(ctx: click.Context) -> None:
    """Start an interactive chat session that uses available MCP tools."""
    def print_events(event: AgentEvent) -> None:
        """Prints events emitted from the FunctionAgent."""
        if isinstance(event, ThinkStartEvent):
            click.echo(colorize_gray(f"\n[{event.iteration + 1}] Thinking..."))
            return
        if isinstance(event, ThoughtEvent):
            click.echo(colorize_gray(f"[{event.iteration + 1}] {event.thought}"))
            if event.tool_name:
                click.echo(colorize_markdown(f"    Function to use: `{event.tool_name}` with args **{event.tool_args}**"))  # noqa: E501
        elif isinstance(event, ToolExecutionStartEvent):
            click.echo(colorize_markdown(f"[{event.iteration + 1}] Executing `{event.tool_name}`..."))  # noqa: E501
        elif isinstance(event, ToolExecutionResultEvent):
            click.echo(colorize_markdown(f"[{event.iteration + 1}] Result from `{event.tool_name}`: =={event.result}=="))  # noqa: E501

    message_history: list[Message] = []
    async with MCPClientManager() as manager:
        await manager.connect_servers(ctx.obj['config_path'])
        mcp_tools = await manager.list_tools()

        if not mcp_tools:
            click.echo("\nNo tools available. Exiting.")
            return
        click.echo("\nAvailable tools:")
        for tool in mcp_tools:
            click.echo(f"- {tool['name']}")

        functions = await convert_mcp_tools_to_functions(mcp_tools, manager)
        assert len(functions) == len(mcp_tools)

        model = ctx.obj['model']
        base_url = ctx.obj['base_url']
        agent = FunctionAgent(
            model_config=ModelConfiguration(
                model=model,
                api_key='None' if base_url else OPENAI_API_KEY,
                base_url=base_url,
            ),
            tools=functions,
            callback=print_events,
        )

        click.echo("\nChat session started. Type 'quit' to exit.")
        while True:
            try:
                user_input = input("\n>> ").strip()
                if user_input.lower() == 'quit':
                    break
                message_history.append(Message(role="user", content=user_input))
                click.echo("\n\nStarting agent...")
                result = await agent(messages=message_history)
                if result.answer:
                    message_history.append(Message(role="assistant", content=result.answer))
                    click.echo(f'\nResponse: "{result.answer}"')
            except Exception as e:
                click.echo(colorize_red(f"\nError during chat: {e}"))


@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start an interactive chat session."""
    asyncio.run(_chat(ctx))


async def _list_tools(ctx: click.Context) -> None:
    """Async implementation of list_tools command."""
    async with MCPClientManager() as manager:
        await manager.connect_servers(ctx.obj['config_path'])
        # list_tools gives tools for all connected servers
        tools = await manager.list_tools()
        for tool in tools:
            print(f"{tool['name']}(...)")
            if tool['description']:
                print(tool['description'])
                print('---')

@cli.command()
@click.pass_context
def list_tools(ctx: click.Context) -> None:
    """List available tools from all connected servers."""
    asyncio.run(_list_tools(ctx))


async def _call_tool(ctx: click.Context, tool: str, args: str) -> None:
    """Async implementation of call_tool command."""
    async with MCPClientManager() as manager:
        await manager.connect_servers(ctx.obj['config_path'])
        result = await manager.call_tool(tool, json.loads(args))
        print(result)

@cli.command()
@click.option('--tool', required=True, help='Tool name')
@click.option('--args', required=True, help='Tool arguments as JSON string')
@click.pass_context
def call_tool(ctx: click.Context, tool: str, args: str) -> None:
    """Call a specific tool."""
    asyncio.run(_call_tool(ctx, tool, args))

def main() -> None:
    """Main entry point."""
    cli(obj={})

if __name__ == '__main__':
    main()
