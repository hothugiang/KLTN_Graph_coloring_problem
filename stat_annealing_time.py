import json
import numpy as np
from collections import defaultdict

# Load the JSON data
for i in [15]:
    file_path = f'Res/final/find_annealing_time_3/n={i}.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Initialize a dictionary to store statistics by annealing_time
    annealing_time_stats = defaultdict(lambda: {
        "p_sol_values": [],
        "constraint_values": [],
        "opt_gap_values": []
    })

    # Process the data
    for entry in data:
        annealing_time = None
        for result in entry["result"][:24]:
            if "annealing_time" in result:
                annealing_time = result["annealing_time"]
            elif "results" in result:
                total_solutions = result["results"]["total_solutions"]
                for res in result["results"]["res"]:
                    p_sol = res["P_sol"]
                    constraint = res["Total_constraint_violations"] / total_solutions
                    opt_gap = res["Best_number_of_colors_used"] - res["ILP_num_colors"]

                    annealing_time_stats[annealing_time]["p_sol_values"].append(p_sol)
                    annealing_time_stats[annealing_time]["constraint_values"].append(constraint)
                    annealing_time_stats[annealing_time]["opt_gap_values"].append(opt_gap)

        # Compute statistics for each annealing_time
        final_stats = {}
        for annealing_time, values in annealing_time_stats.items():
            final_stats[annealing_time] = {
                "P_sol_mean": np.mean(values["p_sol_values"]),
                "P_sol_std": np.std(values["p_sol_values"]),
                "Constraint_mean": np.mean(values["constraint_values"]),
                "Constraint_std": np.std(values["constraint_values"]),
                "Opt_gap_mean": np.mean(values["opt_gap_values"]),
                "Opt_gap_std": np.std(values["opt_gap_values"]),
            }

        # Print the statistics
        print("Statistics by annealing_time:")
        for annealing_time, stats in final_stats.items():
            print(f"\nAnnealing time: {annealing_time}")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")

        # Save the statistics to a file
        output_file = f'Res/final/find_annealing_time_3/stat/{i}.json'
        with open(output_file, 'w') as outfile:
            json.dump(final_stats, outfile, indent=4)
