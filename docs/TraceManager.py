import os
import json
from pathlib import Path

def addToTraceJson(
        addition: dict,
        folder_path: Path = Path(__file__).parent.parent / "docs",
        clear: bool = False
    ) -> None:

    # solve path
    file_path = folder_path / "trace.json"

    if file_path.exists() and clear is False and os.stat(file_path).st_size > 0:
        with open(file_path, "r") as file:
            try:
                cur_trace = json.load(file)
            except json.decoder.JSONDecodeError:
                cur_trace: dict = {}
    else:
        cur_trace: dict = {}

    cur_trace.update(addition)

    with open(file_path, "w") as file:
        json.dump(cur_trace, file, indent= 4)
