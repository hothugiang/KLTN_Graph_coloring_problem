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
import pulp

num_colors = 5

def read_graph_from_file(file_path):
    with open(file_path, "r") as file:
        edge_list = eval(file.read()) 
        graph = nx.Graph(edge_list)
    return graph

graphs = [read_graph_from_file(f"Test/2111/5_{x}.txt") for x in range(1, 10)]

def create_bqm(alpha, beta):
    # bqm lớn
    bqm = dimod.BinaryQuadraticModel({}, {}, 0, 'BINARY')

    # Thêm từng bqm nhỏ vào bqm lớn
    for x in range(1, 10):
        file_name = f"Test/2111/5_{x}.txt"
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

        # H_obj
        for c in range(num_colors):
            bqm.add_variable(f'x_{c}', -1)
        for i in V:
            for c in range(num_colors):
                bqm.add_interaction(f'q_{i}_{x}_{c}', f'x_{c}', 1)
        
    return bqm

embedding_file = "Embedding/embedding_5_graph.json"

def get_embed_size(bqm, outputp):
    best_emb = None
    TT = 0
    token = 'DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397'
    child = DWaveSampler(token=token)
    source_edgelist = list(itertools.chain(bqm.quadratic, ((v, v) for v in bqm.linear)))

    stderr = open(outputp.replace('.json', '_log.txt'), 'w')
    target_edgelist = dimod.child_structure_dfs(child).edgelist
    min_val = 1000000000
    min_len = 100000

    while TT < 1:
        TT += 1
        embedding = minorminer.find_embedding(source_edgelist, target_edgelist, verbose=2, max_no_improvement=20,
                                              random_seed=TT,
                                              chainlength_patience=20)
        var1 = len(embedding.keys())
        if var1 == 0:
            print("Failed ", TT, file=stderr)
            continue
        len_emb = max(map(len, embedding.values()))
        var = sum(len(embedding[node]) for node in embedding)
        print(len_emb, var, file=stderr)
        if len_emb < min_len or (len_emb == min_len and min_val > var):
            min_len = len_emb
            min_val = var
            best_emb = embedding
    print(min_len, min_val, file=stderr)
    
    from dwave.embedding import EmbeddedStructure
    pp = EmbeddedStructure(target_edgelist, best_emb)
    import json
    with open(outputp, "w") as f:
        json.dump(pp, f)
    return pp 


# Chuyển về dạng đỉnh cạnh
def process_result(first_sample, graphs):
    graph_results = {}
    valid = {}
    for graph_index, graph in enumerate(graphs):
        graph_key = f"Graph {graph_index + 1}"
        graph_results[graph_key] = []
        color_assignment = {}
        unique_colors = set()
        valid[graph_key] = "Valid result"
        
        # in ra đỉnh: màu
        for i in range(len(graph.nodes())):
            for c in range(num_colors):
                # Biến p_{i, x, c} = 1: đỉnh i của đồ thị x được tô màu c
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
    return graph_results

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

def main():
    alpha = 1
    beta = 1
    bqm = create_bqm(alpha, beta)

    # Sampler
    # sampler = DWaveSampler(token='DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397')
    # sampler = EmbeddingComposite(DWaveSampler(token='DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397'))
    sampler = SimulatedAnnealingSampler()
    # # Get the Pegasus graph from the sampler
    # pegasus_graph = dnx.pegasus_graph(16)

    # # Plot the hardware Pegasus graph
    # dnx.draw_pegasus(pegasus_graph, node_size=2)
    # plt.show()

    # # Retrieve nodes and edges from the actual hardware graph
    # hardware_nodes = set(sampler.nodelist)
    # hardware_edges = set(tuple(sorted(edge)) for edge in sampler.edgelist)

    # # Generate the ideal P-16 Pegasus graph for comparison
    # ideal_graph = dnx.pegasus_graph(16)
    # ideal_nodes = set(ideal_graph.nodes())
    # ideal_edges = set(tuple(sorted(edge)) for edge in ideal_graph.edges())

    # # Identify missing nodes and edges
    # missing_nodes = ideal_nodes - hardware_nodes
    # missing_edges = ideal_edges - hardware_edges

    # # Print the missing nodes and edges
    # print("Missing Nodes (Faulty/Inaccessible Qubits):")
    # for node in sorted(missing_nodes):
    #     print(node)

    # print("\nMissing Edges (Faulty/Inaccessible Couplers):")
    # for edge in sorted(missing_edges):
    #     print(edge)

    # # Draw the ideal Pegasus graph, highlighting missing nodes and edges
    # plt.figure(figsize=(12, 12))

    # # Draw the full graph in light gray
    # dnx.draw_pegasus(ideal_graph, node_color='lightgray', edge_color='lightgray', node_size=2, width=0.5)

    # # Highlight missing nodes by creating a color map
    # node_colors = ['red' if node in missing_nodes else 'lightgray' for node in ideal_graph.nodes]
    # edge_colors = ['blue' if edge in missing_edges else 'lightgray' for edge in ideal_graph.edges]
    # minorminer.find_embedding()
    # # Redraw with missing nodes and edges highlighted
    # dnx.draw_pegasus(ideal_graph, node_color=node_colors, edge_color=edge_colors, node_size=5, width=1)

    # # Add a title for clarity
    # plt.title('Missing Nodes (Red) and Missing Edges (Blue) in D-Wave Hardware Pegasus Graph')
    # plt.show()
    
    # sampler = SimulatedAnnealingSampler()
    # if not os.path.exists() or os.path.getsize(embedding_file) == 0:
    #     embedding = get_embed_size(bqm, embedding_file)
    #     sampler = EmbeddingComposite(DWaveSampler(token='DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397'))
    # else:
    #     with open(embedding_file, "r") as fff:
    #         embbed = json.load(fff)
    #     embedding = {int(k): v for k, v in embbed.items()}
    #     sampler = FixedEmbeddingComposite(DWaveSampler(token='DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397'), embedding=embedding)
    
    # Tham số
    params = {
        "num_reads": 2000,
        # "annealing_time": 20,
        # "chain_strength": 5
    }
    
    # Giải BQM
    response = sampler.sample(bqm, **params)
    # response = sampler.sample(bqm)

    # Lấy sample đầu tiên
    first_sample = response.first.sample
    first_energy = response.first.energy

    print(first_sample)

    # Chuyển KQ về từng đồ thị
    results = process_result(first_sample, graphs)

    output_data = {
        'param': params,
        'energy': float(first_energy), 
        'info': response.info, 
        'results': {
            'results': results,
            'ILP chromatic number': ilp_graph_coloring(graphs[0]) 
        }
    }

    with open('Res/SA.json', 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Kết quả đã được lưu vào file 'results.json'.")
    print(f"Mức năng lượng: ", first_energy)
    print(response.info)

if __name__ == "__main__":
    main()
