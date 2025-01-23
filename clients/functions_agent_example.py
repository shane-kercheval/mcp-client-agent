"""Example of using the FunctionAgent to call functions."""
import asyncio
import os
import time
from src.functions_agent import (
    AgentEvent,
    Message,
    Function,
    FunctionAgent,
    Parameter,
    ParameterType,
    ModelConfiguration,
    ThinkStartEvent,
    ThoughtEvent,
    ToolExecutionResultEvent,
    ToolExecutionStartEvent,
)
from src.utilities import (
    colorize_blue,
    colorize_gray,
    colorize_green,
    colorize_markdown,
    colorize_orange,
)
from dotenv import load_dotenv


def calculate_sum(numbers: list[float]) -> float:  # noqa: D103
    return sum(numbers)

async def list_directory(path: str = ".") -> list[str]:  # noqa: D103
    return os.listdir(path)

# calculator_tool = dspy.Tool(
#     func=calculate_sum,
#     name="calculate_sum",
#     desc="Calculates the sum of a list of numbers",
#     args={
#         "numbers": (list[float], "List of numbers to sum"),
#     },
# )
# directory_tool = dspy.Tool(
#     func=list_directory,
#     name="list_directory",
#     desc="Lists the contents of a local directory on the user's machine.",
#     args={
#         "path": (str, "Directory path to list (default: current directory)"),
#     },
# )

calculator_function = Function(
    name="calculate_sum",
    description="Calculates the sum of a list of numbers",
    parameters=[
        Parameter(
            name="numbers",
            type=ParameterType.ARRAY,
            required=True,
            description="List of numbers to sum",
        ),
    ],
    # func=calculate_sum,
)

directory_function = Function(
    name="list_directory",
    description="Lists the contents of a local directory on the user's machine.",
    parameters=[
        Parameter(
            name="path",
            type=ParameterType.STRING,
            required=False,  # since it has a default value in the original
            description="Directory path to list (default: current directory)",
        ),
    ],
    func=list_directory,
)


def print_events(event: AgentEvent) -> None:
    """Prints events emitted from the FunctionAgent."""
    if isinstance(event, ThinkStartEvent):
        print(colorize_gray(f"\n[{event.iteration + 1}] Thinking..."))
        return
    if isinstance(event, ThoughtEvent):
        print(colorize_gray(f"[{event.iteration + 1}] {event.thought}"))
        if event.tool_name:
            print(colorize_markdown(f"    Function to use: `{event.tool_name}` with args **{event.tool_args}**"))  # noqa: E501
    elif isinstance(event, ToolExecutionStartEvent):
        print(colorize_markdown(f"[{event.iteration + 1}] Executing `{event.tool_name}`..."))
    elif isinstance(event, ToolExecutionResultEvent):
        print(colorize_markdown(f"[{event.iteration + 1}] Result from `{event.tool_name}`: =={event.result}=="))  # noqa: E501


async def main() -> None:
    """Main function to demonstrate FunctionCalls."""
    load_dotenv()

    # Create predictor with tools
    agent = FunctionAgent(
        model_config=ModelConfiguration(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        tools=[
            calculator_function,
            directory_function,
        ],
        callback=print_events,
    )
    # Test questions
    questions = [
        "What is the sum of 5, 10, and 15?",
        "What files are in the current directory?",
        "What is 100 plus the number of files in the current directory?",
        "What is the current stock price of Apple?",
    ]
    for question in questions:
        print(f"\nQUESTION: `{question}`")

        start_time = time.time()
        result = await agent(messages=[Message(role='user', content=question)])
        elapsed_time = time.time() - start_time

        print(f"\nTime taken: {elapsed_time:.2f} seconds")
        if result.answer:
            print(f"\nANSWER: {result.answer}\n")
            print(f"REASONING: {result.reasoning}")

        print("\nTool Calls (Predictions):")
        for call in result.func_calls:
            print(f"   Name: {colorize_orange(call.func_name)}")
            print(f"   Args: {colorize_blue(call.func_args)}")
            print(f"   Result: {colorize_green(call.func_result)}")
            print(f"   Thought: {colorize_gray(call.thought)}")
            print("   ---")
        print("\n------------------------")


if __name__ == "__main__":
    asyncio.run(main())
