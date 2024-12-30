import json
from collections import defaultdict
import numpy as np
import pandas as pd  # Import pandas for tabular display

# Load the JSON data
file_path = 'Res/final/combine_problem/QA/4/(10, 10, 10, 10).json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize a dictionary to store statistics by graph_id
graph_stats = defaultdict(lambda: {
    "P_sol_values": [],
    "Opt_gap_values": [],
})

# List to store row data for the table
table_data = []

TTS_sum = 0

# Khởi tạo danh sách tổng hợp
P_sol_all = []
Opt_gap_all = []

# Process the data
for entry in data:
    running_time = entry["running_time"]
    print(f"\n Running time: {running_time / 7}")
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

                # Calculate TTS for the current entry
                tts = time_per_sample / (min_p_sol * 10) if min_p_sol != 0 else 0
                TTS_sum += tts

                # Add data to table
                table_data.append({
                    # "Graph ID": graph_id,
                    "Total Solutions": total_solutions,
                    "Time Per Sample (ns)": time_per_sample,
                    "P_sol": p_sol,
                    "Opt Gap": opt_gap,
                    "TTS (ms)": tts,
                })

print(f'Total TTS: {TTS_sum / 4}')

# Calculate final statistics
final_stats = {}
for graph_id, values in graph_stats.items():
    final_stats[graph_id] = {
        "P_sol_mean": np.mean(values["P_sol_values"]),
        "P_sol_std": np.std(values["P_sol_values"]),
        "Opt_gap_mean": np.mean(values["Opt_gap_values"]),
        "Opt_gap_std": np.std(values["Opt_gap_values"]),
    }

# Print aggregate statistics
print("\nAggregate Statistics:")
print(f"P_sol Mean: {np.mean(P_sol_all):.4f}")
print(f"P_sol Std: {np.std(P_sol_all):.4f}")
print(f"Opt_gap Mean: {np.mean(Opt_gap_all):.4f}")
print(f"Opt_gap Std: {np.std(Opt_gap_all):.4f}")

# Create a DataFrame from the table data
df = pd.DataFrame(table_data)

# Export to Excel
excel_file_path = 'graph_stat_same.xlsx'
df.to_excel(excel_file_path, index=False)
print(f"\nTable has been saved to {excel_file_path}")

# Optional: Display the DataFrame
print("\nTable of Results:")
print(df)
