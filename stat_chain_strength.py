import json
import numpy as np
from collections import defaultdict

# Load the JSON data
for i in [5, 10, 15, 20]:
    file_path = f'Res/final/find_chain_strength/n={i}.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Initialize a dictionary to store statistics by chain_strength
    chain_strength_stats = defaultdict(lambda: {
        "p_sol_values": [],
        "constraint_values": [],
        "opt_gap_values": []
    })

    # Process the data
    for entry in data:
        chain_strength = None
        for result in entry["result"][:24]:
            if "chain_strength" in result:
                chain_strength = result["chain_strength"]
            elif "results" in result:
                total_solutions = result["results"]["total_solutions"]
                for res in result["results"]["res"]:
                    p_sol = res["P_sol"]
                    constraint = res["Total_constraint_violations"] / total_solutions
                    opt_gap = res["Best_number_of_colors_used"] - res["ILP_num_colors"]

                    chain_strength_stats[chain_strength]["p_sol_values"].append(p_sol)
                    chain_strength_stats[chain_strength]["constraint_values"].append(constraint)
                    chain_strength_stats[chain_strength]["opt_gap_values"].append(opt_gap)

        # Compute statistics for each chain_strength
        final_stats = {}
        for chain_strength, values in chain_strength_stats.items():
            final_stats[chain_strength] = {
                "P_sol_mean": np.mean(values["p_sol_values"]),
                "P_sol_std": np.std(values["p_sol_values"]),
                "Constraint_mean": np.mean(values["constraint_values"]),
                "Constraint_std": np.std(values["constraint_values"]),
                "Opt_gap_mean": np.mean(values["opt_gap_values"]),
                "Opt_gap_std": np.std(values["opt_gap_values"]),
            }

        # Print the statistics
        print("Statistics by chain_strength:")
        for chain_strength, stats in final_stats.items():
            print(f"\nChain Strength: {chain_strength}")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")

        # Save the statistics to a file
        output_file = f'Res/final/find_chain_strength/stat2/{i}.json'
        with open(output_file, 'w') as outfile:
            json.dump(final_stats, outfile, indent=4)
