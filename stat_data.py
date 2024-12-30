import json
from collections import defaultdict
import numpy as np
import pandas as pd  # Import pandas for tabular display

# Load the JSON data
file_path = 'Res/final/combine_problem/QA/1/n=10.json'
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

# Process the data
for entry in data:
    running_time = entry["running_time"]
    print(f"\n Running time: {running_time / 30}")
    for result in entry["result"][:24]:
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
                if p_sol < min_p_sol:
                    min_p_sol = p_sol
                opt_gap = res["Best_number_of_colors_used"] - res["ILP_num_colors"]

                # Calculate TTS for the current entry
                tts = time_per_sample / (min_p_sol * 10) if min_p_sol != 0 else 0
                TTS_sum += tts

                graph_stats[graph_id]["P_sol_values"].append(p_sol)
                graph_stats[graph_id]["Opt_gap_values"].append(opt_gap)

                # Add data to table
                table_data.append({
                    "Total Solutions": total_solutions,
                    "Time Per Sample (ns)": time_per_sample,
                    "P_sol": p_sol,
                    "Opt Gap": opt_gap,
                    "TTS (ms)": tts,
                })

print(f'Total TTS: {TTS_sum}')

# Calculate final statistics
final_stats = {}
for graph_id, values in graph_stats.items():
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

# Create a DataFrame from the table data
df = pd.DataFrame(table_data)

# Export to Excel
excel_file_path = 'graph_10.xlsx'
df.to_excel(excel_file_path, index=False)
print(f"\nTable has been saved to {excel_file_path}")

# Optional: Display the DataFrame
print("\nTable of Results:")
print(df)
