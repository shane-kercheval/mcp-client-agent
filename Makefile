.PHONY: tests app build


# view logs: tail -n 20 -F ~/Library/Logs/Claude/mcp*.log

####
# Project
####
build:
	uv sync

install_fake_server:
	# see readme for information on errors when starting claude desktop
	uv run mcp install ./servers/mcp_fake_server.py

chat:
	PYTHONPATH=$PYTHONPATH:. uv run ./clients/mcp_client.py \
		--config ./servers/mcp_fake_server_config.json \
		--model 'openai/gpt-4o-mini' \
		chat

chat_4o__config_1:
	PYTHONPATH=$PYTHONPATH:. uv run ./clients/mcp_client.py \
		--config ~/Library/Application\ Support/Claude/claude_desktop_config.json \
		--model 'openai/gpt-4o-mini' \
		chat

chat_nav:
	PYTHONPATH=$PYTHONPATH:. uv run ./clients/mcp_client.py \
		--config ./servers/mcp_fake_server_config.json \
		--base_url 'http://127.0.0.1:8080' \
		chat

chat_local:
	PYTHONPATH=$PYTHONPATH:. uv run ./clients/mcp_client.py \
		--config ./servers/mcp_fake_server_config.json \
		--base_url 'http://127.0.0.1:1234/v1' \
		chat

run_agent_example:
	PYTHONPATH=$PYTHONPATH:. uv run ./clients/functions_agent_example.py

run_cli_list_tools:
	PYTHONPATH=$PYTHONPATH:. uv run ./clients/mcp_client.py \
		--config ./servers/mcp_fake_server_config.json \
		list-tools

run_cli_call_tool_calc:
	PYTHONPATH=$PYTHONPATH:. uv run ./clients/mcp_client.py \
		--config ./servers/mcp_fake_server_config.json \
		call-tool \
		--tool calculator_sum \
		--args '{"numbers": [1, 2, 3]}'

inspect:
	# uv run mcp dev src/server.py
	PYTHONPATH=$PYTHONPATH:. \
		SERVER_PORT=9000 \
		npx @modelcontextprotocol/inspector \
		uv \
		run \
		./servers/mcp_fake_server.py

linting:
	uv run ruff check src
	uv run ruff check clients
	uv run ruff check servers
	uv run ruff check tests

tests: linting
	# uv run pytest tests
	PYTHONPATH=$PYTHONPATH:. uv run pytest tests
