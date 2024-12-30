import json
from collections import defaultdict
import numpy as np

# Load the JSON data
file_path = 'Res/final/combine_problem/QA/4/(20, 20, 20, 20).json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize a dictionary to store statistics by graph_id
graph_stats = defaultdict(lambda: {
    "P_sol_values": [],
    "Opt_gap_values": [],
})

tts_values = []
TTS_sum = 0

# Khởi tạo danh sách tổng hợp
P_sol_all = []
Opt_gap_all = []

# Process the data
for entry in data:
    running_time = entry["running_time"]
    print(f"\n Running time: {running_time/7}")
    for result in entry["result"][:6]:
        if "results" in result:
            total_solutions = result["results"]["total_solutions"]
            info = result["results"]["info"]
            if "qpu_access_time" in info["timing"]:
                time_per_sample = info["timing"]["qpu_access_time"] / total_solutions
            elif all(k in info["timing"] for k in ["preprocessing_ns", "sampling_ns", "postprocessing_ns"]):
                time_per_sample = (
                    info["timing"]["preprocessing_ns"] +
                    info["timing"]["sampling_ns"] +
                    info["timing"]["postprocessing_ns"]
                ) / total_solutions

            min_p_sol = 100
            for res in result["results"]["res"]:
                graph_id = res["graph_id"]
                p_sol = res["P_sol"]
                if (p_sol < min_p_sol): min_p_sol = p_sol
                opt_gap = res["Best_number_of_colors_used"] - res["ILP_num_colors"]

                # Thêm giá trị vào graph_stats
                graph_stats[graph_id]["P_sol_values"].append(p_sol)
                graph_stats[graph_id]["Opt_gap_values"].append(opt_gap)

                # Thêm giá trị vào danh sách tổng hợp
                P_sol_all.append(p_sol)
                Opt_gap_all.append(opt_gap)

            if (min_p_sol != 0): 
                TTS_sum += time_per_sample / (min_p_sol / 100)

print(f'TTS: {TTS_sum / 4000}')

# Tính toán thống kê cho từng graph_id
final_stats = {}
for graph_id, values in graph_stats.items():
    final_stats[graph_id] = {
        "P_sol_mean": np.mean(values["P_sol_values"]),
        "P_sol_std": np.std(values["P_sol_values"]),
        "Opt_gap_mean": np.mean(values["Opt_gap_values"]),
        "Opt_gap_std": np.std(values["Opt_gap_values"]),
    }

# In thống kê tổng hợp
print("\nDanh sách tổng hợp P_sol và Opt_gap:")
print(f"P_sol: {np.mean(P_sol_all)}")
print(f"P_sol_std: {np.std(P_sol_all)}")
print(f"Opt_gap: {np.mean(Opt_gap_all)}")
print(f"Opt_gap_std: {np.std(Opt_gap_all)}")
