"""Simple fake FastMCP server."""
from mcp.server.fastmcp import FastMCP
# Initialize FastMCP server
mcp = FastMCP("fake-server")

@mcp.tool()
async def reverse_text(text: str) -> str:
    """
    Reverse the input text.

    Args:
        text: The text to reverse
    """
    return text[::-1]

@mcp.tool()
async def calculator_sum(numbers: list[float]) -> float:
    """
    Calculate the sum of a list of numbers.

    Args:
        numbers: List of numbers to sum
    """
    return sum(numbers)

@mcp.tool()
async def calculator_multiply(numbers: list[float]) -> float:
    """
    Calculate the multiplication of a list of numbers (e.g., 2 * 3 * 4 = 24).

    Args:
        numbers: List of numbers to multiply together.
    """
    result = 1
    for num in numbers:
        result *= num
    return result

@mcp.tool()
async def count_words(text: str) -> int:
    """
    Count the number of words in a text.

    Args:
        text: The text to analyze
    """
    return len(text.split())

@mcp.tool()
async def count_characters(text: str) -> int:
    """
    Count the number of words in a text.

    Args:
        text: The text to analyze
    """
    return len(text)


# # Global namespace to maintain state between executions
# GLOBAL_NAMESPACE: dict[str, object] = {'__builtins__': __builtins__}

# @mcp.tool()
# async def execute_python_code(code: str) -> str:
#     """
#     Execute Python code and return the output. Maintains state between executions.

#     If an error occurs, the output will contain the error message.

#     Args:
#         code: Python code to execute as a string
#     """
#     output = io.StringIO()
#     error = None
#     try:
#         # Redirect stdout/stderr to capture all output
#         with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
#             exec(code, GLOBAL_NAMESPACE)
#     except Exception as e:
#         error = str(e)
#     finally:
#         # Get the captured output
#         output_value = output.getvalue()
#         output.close()

#     if error:
#         return f"Error: {error}\nOutput: {output_value}"

#     return output_value


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
