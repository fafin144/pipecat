"""Minimal voice-friendly agent: Claude + a weather tool (Open-Meteo, no API
key required). Invoked when a caller says "Hey AI Agent" on the bridge call.

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from anthropic import AsyncAnthropic

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a named city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'London'"},
        },
        "required": ["city"],
    },
}


async def _geocode(city: str) -> tuple[float, float, str] | None:
    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
        )
        resp.raise_for_status()
        results = (resp.json() or {}).get("results") or []
    if not results:
        return None
    r = results[0]
    return r["latitude"], r["longitude"], r.get("name", city)


async def _get_weather(city: str) -> str:
    geo = await _geocode(city)
    if geo is None:
        return f"I couldn't find a location called {city}."
    lat, lon, resolved_name = geo
    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"},
        )
        resp.raise_for_status()
        data = resp.json()
    current = data.get("current_weather") or {}
    temp = current.get("temperature")
    wind = current.get("windspeed")
    if temp is None:
        return f"I couldn't get the current weather for {resolved_name}."
    return f"It's currently {temp} degrees Celsius in {resolved_name}, with wind speed {wind} kilometers per hour."


async def _run_tool(name: str, tool_input: dict[str, Any]) -> str:
    if name == "get_weather":
        return await _get_weather(str(tool_input.get("city", "")))
    return f"Unknown tool: {name}"


async def ask_agent(question: str, max_tool_rounds: int = 3) -> str:
    """Send `question` to Claude with the weather tool available, run the
    tool-use loop, and return a short, spoken-friendly final answer."""
    client = _get_client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    for _ in range(max_tool_rounds):
        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            system=(
                "You are a helpful voice assistant answering a question spoken "
                "during a phone call. Keep answers short, spoken-friendly, and "
                "conversational - one or two sentences, no markdown or lists."
            ),
            tools=[WEATHER_TOOL],
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            text = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            return text or "Sorry, I don't have an answer for that."

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_text = await _run_tool(block.name, block.input or {})
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
                )
        messages.append({"role": "user", "content": tool_results})

    return "Sorry, I couldn't get an answer in time."
