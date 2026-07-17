import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from docs.TraceManager import loadTraceJson
from eval.EvaluationAgent import EvalAgent

import pandas as pd

def evaluateFromTrace(
        file: str = "trace.json",
        treshold: float = 0.7,
    ):

    # get name and assert typing
    if Path(file).suffix == ".json":
        trace_name = file[:-5]
    elif Path(file).suffix == "":
        trace_name = file
        file+=".json"
    else:
        raise f"INVALID TRACE TYPE({file}) ONLY JSON OR NONE"

    # load contents from trace and feed to judge
    trace = loadTraceJson(file_name= file)

    if trace["query_validation"]["status"] != "PROCEED":
        raise "only 'PROCEED' pipelines can be evaluated"

    context = trace["retrieved_chunks"]
    if "formatted_tool_message" in trace:
        context += f"\n{trace["formatted_tool_message"]["content"]}"

    judge = EvalAgent()
    results: dict = judge.evaluate(
        question= trace["reformulated query"],
        context= context,
        response = trace["main_agent_response"]
    )

    # format response to a dataframe format
    scores_df_format = {
        "metric":[],
        "score":[],
        "status":[]
    }
    for metric in judge.metrics_used:
        scores_df_format["metric"].append(metric)
        scores_df_format["score"].append(results[metric])

        status = "passed" if results[metric] >= treshold else "failed"
        scores_df_format["status"].append(status)

        print(f"{metric}: {results[metric]} --> {status}")

    # exports result as .csv
    exportToCSV(results_file_name= f"{trace_name}(eval).csv", scores= scores_df_format)

def exportToCSV(
        results_file_name: str,
        scores: dict,
        results_folder: Path = Path(__file__).parent
    ) -> None:
        results_path = results_folder / results_file_name
        count = 1

        # avoid clashing into an existing csv
        while results_path.exists():
            new_results_file_name = f"{results_file_name}({count}).csv"
            results_path = results_folder / new_results_file_name
            count += 1

        # export
        results_df = pd.DataFrame(scores)
        results_df.to_csv(path_or_buf=results_path, index=False)

        print(f"successfully results exported into: {results_file_name}")

if __name__ == "__main__" and sys.argv[1]:
    evaluateFromTrace(str(sys.argv[1]))






