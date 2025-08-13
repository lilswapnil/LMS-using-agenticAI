from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Callable, List

@dataclass
class Tool:
    name: str
    fn: Callable[[Dict[str, Any]], str]
    desc: str

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    def call(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].fn(args)
    def spec(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "desc": t.desc} for t in self._tools.values()]
