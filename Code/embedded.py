import json
import dimod
import itertools
import minorminer
import os
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import dwave.embedding
import pulp

num_colors = 5

def read_graph_from_file(file_path):
    with open(file_path, "r") as file:
        edge_list = eval(file.read()) 
        graph = nx.Graph(edge_list)
    return graph

graphs = [read_graph_from_file(f"./../Test/final/5_{x}.txt") for x in range(1, 10)]

def create_bqm(alpha, beta):
    # bqm lớn
    bqm = dimod.BinaryQuadraticModel({}, {}, 0, 'BINARY')

    # Thêm từng bqm nhỏ vào bqm lớn
    for x in range(1, 2):
        file_name = f"./../Test/final/5_{x}.txt"
        graph = read_graph_from_file(file_name)

        V = list(graph.nodes())
        E = list(graph.edges())

        # H1: Ràng buộc mỗi đỉnh chỉ được tô duy nhất một màu
        for i in V:
            for c in range(num_colors):
                bqm.add_variable(f'q_{i}_{x}_{c}', -alpha)
            # Thêm chi phí năng lượng tăng nếu một đỉnh có nhiều hơn một màu
            for c in range(num_colors - 1):
                for c_prime in range(c + 1, num_colors):
                    bqm.add_interaction(f'q_{i}_{x}_{c}', f'q_{i}_{x}_{c_prime}', 2 * alpha)
            bqm.offset += alpha

        # H2: Ràng buộc không có 2 đỉnh kề nào cùng màu
        for (i, j) in E:
            for c in range(num_colors):
                bqm.add_interaction(f'q_{i}_{x}_{c}', f'q_{j}_{x}_{c}', beta)
    
    return bqm

def main():
    alpha = 1
    beta = 1
    bqm = create_bqm(alpha, beta)

    # Embedding
    token = 'DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397'
    hardware_sampler = DWaveSampler(token=token)
    hardware_edges = hardware_sampler.edgelist

    source_edgelist = list(bqm.quadratic.keys())
    embedding = minorminer.find_embedding(source_edgelist, hardware_edges)

    if not embedding:
        print("Không thể tìm thấy embedding phù hợp.")
        return

    # embeded BQM
    embedded_bqm = dwave.embedding.embed_bqm(bqm, embedding, nx.Graph(hardware_edges), chain_strength=5.0)

    print(embedding)
    # sampler
    sampler = SimulatedAnnealingSampler()
    # sampler = FixedEmbeddingComposite(DWaveSampler(token=token), embedding)
    embedded_sampleset = sampler.sample(embedded_bqm, num_reads=1000)

    # result
    unembedded_sampleset = dwave.embedding.unembed_sampleset(embedded_sampleset, embedding, bqm)
    first_sample = unembedded_sampleset.first.sample
    first_energy = unembedded_sampleset.first.energy

    results, res = process_result(first_sample, graphs)

    output_data = {
        'param': {"num_reads": 1000},
        'energy': float(first_energy),
        'info': embedded_sampleset.info,
        'results': {
            'results': results,
            'res': res,
        }
    }

    # with open('Res/SA.json', 'w') as f:
    #     json.dump(output_data, f, indent=4)

    print(f"Kết quả đã được lưu vào file 'SA.json'.")
    print(f"Mức năng lượng: ", first_energy)
    print(embedded_sampleset.info)

    # Bước 5: Visualize Pegasus graph
    # Retrieve nodes and edges from the actual hardware graph
    hardware_nodes = set(hardware_sampler.nodelist)
    hardware_edges = set(tuple(sorted(edge)) for edge in hardware_sampler.edgelist)

    # Generate the ideal P-16 Pegasus graph for comparison
    ideal_graph = dnx.pegasus_graph(16)
    ideal_nodes = set(ideal_graph.nodes())
    ideal_edges = set(tuple(sorted(edge)) for edge in ideal_graph.edges())

    # Identify missing nodes and edges
    missing_nodes = ideal_nodes - hardware_nodes
    missing_edges = ideal_edges - hardware_edges

    # Print the missing nodes and edges
    print("Missing Nodes (Faulty/Inaccessible Qubits):")
    for node in sorted(missing_nodes):
        print(node)

    print("\nMissing Edges (Faulty/Inaccessible Couplers):")
    for edge in sorted(missing_edges):
        print(edge)

    # Draw the ideal Pegasus graph, highlighting missing nodes and edges
    plt.figure(figsize=(12, 12))

    # Draw the full graph in light gray
    dnx.draw_pegasus(ideal_graph, node_color='lightgray', edge_color='lightgray', node_size=2, width=0.5)

    # Highlight missing nodes by creating a color map
    used_nodes = []
    chains = []
    inter_chain_couplers = []
    for logical_qubit, physical_qubits in embedding.items():
        used_nodes.extend(physical_qubits)
        chain_edges = [(physical_qubits[i], physical_qubits[i + 1]) for i in range(len(physical_qubits) - 1)]
        chains.extend(chain_edges)

        # Identify couplers connecting different chains
        for physical_qubit in physical_qubits:
            for neighbor in ideal_graph.neighbors(physical_qubit):
                if neighbor not in physical_qubits and neighbor in used_nodes:
                    inter_chain_couplers.append((physical_qubit, neighbor))

    node_colors = []
    for node in ideal_graph.nodes:
        if node in missing_nodes:
            node_colors.append('red')
            continue
        if node in used_nodes:
            node_colors.append('green')
            continue
        node_colors.append('lightgray')
    edge_colors = []
    for edge in ideal_graph.edges:
        if edge in missing_edges:
            edge_colors.append('red')
            continue
        if edge in chains:
            edge_colors.append('green')
            continue
        if edge in inter_chain_couplers:
            edge_colors.append('blue')
            continue
        edge_colors.append('lightgray')

    # Redraw with missing nodes and edges highlighted
    dnx.draw_pegasus(ideal_graph, node_color=node_colors, edge_color=edge_colors, node_size=5, width=1)

    # Add a title for clarity
    plt.title('Missing Nodes (Red), Intra-Chain Couplers (Green), Inter-Chain Couplers (Blue) in D-Wave Hardware Pegasus Graph')
    plt.show()

def process_result(first_sample, graphs):
    graph_results = {}
    valid = {}
    pass_res = {}

    for graph_index, graph in enumerate(graphs):
        graph_key = f"Graph {graph_index + 1}"
        graph_results[graph_key] = []
        color_assignment = {}
        unique_colors = set()
        valid[graph_key] = "Valid result"
        pass_res[graph_key] = 0
        
        # in ra đỉnh: màu
        for i in range(len(graph.nodes())):
            for c in range(num_colors):
                key = f'q_{i}_{graph_index + 1}_{c}'
                if first_sample.get(key) == 1:
                    graph_results[graph_key].append(f"{i}:{c}")
                    color_assignment[i] = c 
                    unique_colors.add(c)

        # Kiểm tra điều kiện "mỗi đỉnh chỉ được tô một màu"
        for i in range(len(graph.nodes())):
            assigned_colors = [c for c in range(num_colors) if first_sample.get(f'q_{i}_{graph_index + 1}_{c}') == 1]
            if len(assigned_colors) != 1:
                valid[graph_key] = f"H1: Invalid (Vertex {i} has multiple or no colors)"
                break

        # Kiểm tra điều kiện "các đỉnh kề nhau không được cùng màu"
        for (i, j) in graph.edges():
            if color_assignment.get(i) == color_assignment.get(j):
                valid[graph_key] = f"H2: Invalid (Adjacent vertices {i} and {j} have the same color)"
                break
    
        graph_results[graph_key].append(valid[graph_key])
        graph_results[graph_key].append(f"Chromatic number: {len(unique_colors)}")

        # Kiểm tra nếu kết quả hợp lệ và số màu là tối ưu
        if valid[graph_key] == "Valid result" and len(unique_colors) == ilp_graph_coloring(graph):
            pass_res[graph_key] = 1

    return graph_results, pass_res


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
        problem += pulp.lpSum(x[i][v] for i in colors) == 1  # Each vertex must have exactly one color
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

if __name__ == "__main__":
    main()
