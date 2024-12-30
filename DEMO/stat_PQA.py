import json
from collections import defaultdict
import numpy as np

# Load the JSON data
file_path = 'Res/(10, 15).json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize a dictionary to store statistics by graph_id
graph_stats = defaultdict(lambda: {
    "P_sol_values": [],
    "Opt_gap_values": [],
})

tts_values = []
TTS_sum = 0

# Process the data
for entry in data:
    running_time = entry["running_time"]
    print(f"\n Running time: {running_time}")
    for result in entry["result"]:
        if "results" in result:
            total_solutions = result["results"]["total_solutions"]
            time_per_sample = result["results"]["Time_per_sample"]
            min_p_sol = 100
            for res in result["results"]["res"]:
                graph_id = res["graph_id"]
                p_sol = res["P_sol"]
                if (p_sol < min_p_sol): min_p_sol = p_sol
                opt_gap = res["Best_number_of_colors_used"] - res["ILP_num_colors"]

                graph_stats[graph_id]["P_sol_values"].append(p_sol)
                graph_stats[graph_id]["Opt_gap_values"].append(opt_gap)
            # print(min_p_sol)
            if (min_p_sol != 0): TTS_sum += time_per_sample / min_p_sol

print(f'TTS: {TTS_sum}')

final_stats = {}
for graph_id, values in graph_stats.items():
    if np.mean(values["P_sol_values"]) < min_p_sol: min_p_sol = np.mean(values["P_sol_values"])
    final_stats[graph_id] = {
        "P_sol_mean": np.mean(values["P_sol_values"]),
        "P_sol_std": np.std(values["P_sol_values"]),
        "Opt_gap_mean": np.mean(values["Opt_gap_values"]),
        "Opt_gap_std": np.std(values["Opt_gap_values"]),
    }

# Print the statistics by graph_id
for graph_id, stats in final_stats.items():
    print(f"\nGraph ID: {graph_id}")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
