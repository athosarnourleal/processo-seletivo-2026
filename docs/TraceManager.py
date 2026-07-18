import os
import json
from pathlib import Path

def addToTraceJson(
        addition: dict,
        folder_path: Path = Path(__file__).parent.parent / "docs",
        clear: bool = False
    ) -> None:

    file_path = folder_path / "trace.json"

    cur_trace:dict = {}
    if not clear:
        cur_trace = loadTraceJson(folder_path=folder_path, file_name="trace.json")

    cur_trace.update(addition)

    with open(file_path, "w") as file:
        json.dump(cur_trace, file, indent= 4)

def loadTraceJson(
        folder_path: Path = Path(__file__).parent.parent / "docs",
        file_name: str = "trace.json"
    ):

    file_path = folder_path / file_name

    if isTraceValid(folder_path= folder_path, file_name= file_name):
        with open(file_path, "r") as file:
            try:
                cur_trace = json.load(file)
            except json.decoder.JSONDecodeError:
                cur_trace = {}

        return cur_trace
    else:
        return {}

def isTraceValid(
        folder_path: Path = Path(__file__).parent.parent / "docs",
        file_name: str = "trace.json"
) -> bool:
    file_path = folder_path / file_name

    if file_path.exists() and os.stat(file_path).st_size > 0:
        return True
    else:
        return False
