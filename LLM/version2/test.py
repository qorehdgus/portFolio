from mcp.server.fastmcp import FastMCP


# Create an MCP server
mcp = FastMCP("mcp_project")


# Add an additional tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def introduce_person(a: str) -> str:
    """Introduce a specific person"""
    return f"{a} is a good person!"

# Add a dynamic greeing resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run()