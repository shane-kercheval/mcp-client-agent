# MCP-Agent

`./clients/mcp_client.py` is a CLI that uses DSPy to interact with MCP servers. The goal is to provide a similar experience to Claude Desktop. In particular, the DSPy agent uses the tool definitions defined in the MCP server(s) to autonomously and iteratively use the tools to accomplish the user's request.

The CLI program takes a configuration file in the same format that Claude Desktop uses and loads all of the servers in the config. The tools across all servers are available for the agent to use.

# Running the Project

Useful commands for running the project can be found in the `Makefile`.

## Quickstart

- create a `.env` file in the project directory and add `OPENAI_API_KEY` key and value.
- `make build`
- `make chat` 
    - starts the `chat` CLI command using the `./servers/mcp_fake_server_config.json` config to start the MCP Server `./servers/mcp_fake_server.py`
        - This MCP server contains trivial tools like `count_characters` or `calulator_multiply`
    - this command uses `openai/gpt-4o-mini` as the model (DSPy model path formatting) and requires `OPENAI_API_KEY` to be set in the `.env` file

Example using `make chat` command and `mcp_fake_server.py`:

> Multiply the number of characters in both the following two texts: "Lorem ipsum odor amet, consectetuer adipiscing elit. Nisl ligula molestie nec a sodales morbi vel. Diam interdum metus ante semper, lorem ornare massa. Accumsan dapibus lacus venenatis; elementum varius nascetur nibh sodales ipsum. Suspendisse mauris cubilia sit elit netus mattis. Rhoncus vivamus ridiculus nostra sociosqu bibendum. Eleifend ultricies orci proin maecenas justo felis sagittis. Elit blandit mattis dis consectetur morbi quisque penatibus." and "Lorem ipsum odor amet, consectetuer adipiscing elit. Proin maximus condimentum fusce nam, dictum adipiscing maximus. Lacinia pellentesque magna senectus adipiscing massa cras elementum tortor. Habitant hendrerit sodales, consectetur amet conubia ullamcorper. Suscipit aliquet curae lobortis imperdiet, montes lectus. Hendrerit penatibus commodo semper laoreet viverra eu elit nisl."

The `count_characters` and `calculator_multiply` tools should be used and result should be `455 * 381 = 173,355`.

## Error When Starting Claude

I received these errors when starting Claude after installing the server via `uv run mcp install ./servers/mcp_fake_server.py`. These errors do not happen when using docker containers. One isue is that the `uv` command doesn't seem to be found unless I provide the path (found via `which uv`). Another issue is that the MCP server cannot find files from different directories that it attempts to import.

Here are the errors I get (`tail -n 20 -F ~/Library/Logs/Claude/mcp*.log` helped me to debug):

```
Error in MCP connection to server Demo: Error: spawn uv ENOENT
[error] Could not start MCP server Demo: Error: spawn uv ENOENT
```

The solution was to add the full path to the `command` and add the `--directory` flag referencing the project directory of the source files that are imported by the server:

```
{
  "mcpServers": {
    "fake-server": {
      "command": "/Users/shanekercheval/.local/bin/uv",
      "args": [
        "run",
        "--directory", 
        "/Users/shanekercheval/repos/mcp-local-files",
        "--with",
        "mcp",
        "mcp",
        "run",
        "servers/mcp_fake_server.py"
      ]
    }
  }
}
```

There doesn't appear to be a way to do this directly from the install command.
