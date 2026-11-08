from fastmcp import FastMCP

mcp = FastMCP("Weather")

SAMPLE_WEATHER = {
    "paris": "18C, cloudy",
    "tokyo": "24C, sunny",
    "dhaka": "31C, humid",
}


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city. Uses sample data, not live data."""
    return SAMPLE_WEATHER.get(city.strip().lower(), "No data for that city.")


if __name__ == "__main__":
    mcp.run(transport="stdio")