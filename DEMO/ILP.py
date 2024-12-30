import os
import networkx as nx
import pulp
import json

def read_graph_from_file(file_path):
    with open(file_path, "r") as file:
        edge_list = eval(file.read())
        graph = nx.Graph(edge_list)
    return graph

def ilp_graph_coloring(graph):
    problem = pulp.LpProblem("GraphColoring", pulp.LpMinimize)

    # Variables
    colors = range(1, len(graph) + 1)  
    vertices = list(graph.nodes())  

    x = pulp.LpVariable.dicts("x", (colors, vertices), 0, 1, pulp.LpInteger)
    w = pulp.LpVariable.dicts("w", colors, 0, 1, pulp.LpInteger)

    # Objective function: minimize the number of colors used
    problem += pulp.lpSum(w[i] for i in colors)

    # Constraints
    for v in vertices:
        problem += pulp.lpSum(x[i][v] for i in colors) == 1 
        for i in colors:
            problem += x[i][v] <= w[i]  # If a color is used for a vertex, set w[i] = 1

    for u in vertices:
        for v in graph[u]:
            if u < v:  # Ensure each edge constraint is only added once
                for i in colors:
                    problem += x[i][u] + x[i][v] <= 1  # Adjacent vertices cannot have the same color

    problem.solve()

    # Find the minimum number of colors used
    min_colors_used = sum(1 for i in colors if pulp.value(w[i]) == 1)

    return min_colors_used

def main():
    results = []

    for i in [5, 10, 15, 20]:
        for j in range(1, 3):
            filename = f"{i}_{j}.txt"
            file_path = f"Test/{i}_{j}.txt"
            graph = read_graph_from_file(file_path)
            num_edges = graph.number_of_edges()
            min_colors = ilp_graph_coloring(graph)
            results.append({
                "graph": filename,
                "num_edges": num_edges,
                "min_colors": min_colors
            })

    # Save results to a file
    with open('Res/input_stat.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    main()
