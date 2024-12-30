import json
import dimod
import itertools
import minorminer
import os
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite
import dwave.embedding
import pulp
import concurrent.futures
import time

num_colors = 10

def read_graph_from_file(file_path):
    with open(file_path, "r") as file:
        edge_list = eval(file.read()) 
        graph = nx.Graph(edge_list)
    return graph

def create_bqm_for_graph(graph, alpha, beta, graph_id):
    bqm = dimod.BinaryQuadraticModel({}, {}, 0, 'BINARY')
    V = list(graph.nodes())
    E = list(graph.edges())

    # H1: Ràng buộc mỗi đỉnh chỉ được tô duy nhất một màu
    for i in V:
        for c in range(num_colors):
            bqm.add_variable(f'q_{i}_{graph_id}_{c}', -alpha)
        # Thêm chi phí năng lượng tăng nếu một đỉnh có nhiều hơn một màu
        for c in range(num_colors - 1):
            for c_prime in range(c + 1, num_colors):
                bqm.add_interaction(f'q_{i}_{graph_id}_{c}', f'q_{i}_{graph_id}_{c_prime}', 2 * alpha)
        bqm.offset += alpha

    # H2: Ràng buộc không có 2 đỉnh kề nào cùng màu
    for (i, j) in E:
        for c in range(num_colors):
            bqm.add_interaction(f'q_{i}_{graph_id}_{c}', f'q_{j}_{graph_id}_{c}', beta)
    
    return bqm

def find_embeddings(bqm_list, hardware_edges):
    embeddings = []
    available_edges = set(hardware_edges)  # Tập các cạnh khả dụng

    for idx, bqm in enumerate(bqm_list):
        available_graph = nx.Graph()
        available_graph.add_edges_from(available_edges)

        # Lấy danh sách các cạnh từ BQM
        source_edgelist = list(bqm.quadratic.keys())

        embedding = minorminer.find_embedding(source_edgelist, available_graph.edges())

        if embedding:
            embeddings.append(embedding)

            # Cập nhật các cạnh khả dụng sau khi sử dụng
            used_qubits = {qubit for qubits in embedding.values() for qubit in qubits}
            available_edges = {
                edge for edge in available_edges
                if not (edge[0] in used_qubits or edge[1] in used_qubits)
            }
        else:
            print(f"Không thể tìm thấy embedding phù hợp cho bài toán {idx + 1}.")

    return embeddings

def sample_bqm(bqm_combined, sampler, num_reads=1000):
    start_time_annealing = time.time()
    sampleset = sampler.sample(bqm_combined, num_reads=num_reads)
    end_time_annealing = time.time()
    annealing_time = end_time_annealing - start_time_annealing

    result = {
        "samples": dict(sampleset.first.sample),
        "annealing_time": annealing_time,
        "info": sampleset.info
    }
    print(f"Annealing time: {annealing_time} seconds")
    return result

def parallel_annealing(alpha, beta, token, graphs, chain_strength):
    bqm_list = [create_bqm_for_graph(graph, alpha, beta, graph_id) for graph_id, graph in enumerate(graphs)]
    
    hardware_sampler = DWaveSampler(token=token)
    hardware_edges = hardware_sampler.edgelist
    
    embeddings = find_embeddings(bqm_list, hardware_edges)
    
    if not embeddings:
        print("Không thể tìm thấy embedding phù hợp cho tất cả các bài toán.")
        return None, None

    # Kết hợp tất cả BQM thành một BQM duy nhất
    bqm_combined = dimod.BinaryQuadraticModel({}, {}, 0, 'BINARY')
    for bqm, (embedding, _) in zip(bqm_list, embeddings):
        embedded_bqm = dwave.embedding.embed_bqm(bqm, embedding, nx.Graph(hardware_sampler.edgelist), chain_strength=chain_strength)
        bqm_combined.update(embedded_bqm)

    # Thực hiện ủ cho BQM kết hợp bằng SimulatedAnnealingSampler
    result = sample_bqm(bqm_combined, SimulatedAnnealingSampler())
    
    return result, [embedding for embedding, _ in embeddings]

def interpret_combined_results(result, graphs):
    interpreted_results = []
    for graph_id, graph in enumerate(graphs):
        node_color_mapping = {}
        
        # Lấy màu từ kết quả cho từng bài toán con
        for key, value in result['samples'].items():
            if value == 1:
                parts = key.split('_')
                if len(parts) == 4:
                    node_id = int(parts[1])
                    current_graph_id = int(parts[2])
                    color = int(parts[3])
                    if current_graph_id == graph_id:
                        node_color_mapping[node_id] = color

        interpreted_results.append({
            "graph_id": graph_id,
            "node_color_mapping": node_color_mapping
        })

    return interpreted_results

def check_graph_coloring_result(interpreted_results, graphs):
    check_results = []

    for result in interpreted_results:
        graph_id = result['graph_id']
        node_color_mapping = result['node_color_mapping']
        graph = graphs[graph_id]

        # Condition 1: Each vertex should have exactly one color
        valid = True
        constraint_violations = 0

        for node in graph.nodes():
            if node not in node_color_mapping:
                valid = False
                constraint_violations += 1
            else:
                assigned_colors = [color for n, color in node_color_mapping.items() if n == node]
                if len(assigned_colors) != 1:
                    valid = False
                    constraint_violations += 1

        # Condition 2: No two adjacent vertices should have the same color
        if valid:
            for u, v in graph.edges():
                if node_color_mapping.get(u) == node_color_mapping.get(v):
                    valid = False
                    constraint_violations += 1

        check_results.append({
            "graph_id": graph_id,
            "valid": int(valid),
            "constraint_violations": constraint_violations
        })

    return check_results

def main(graphs, chain_strength):
    alpha = 1
    beta = 1
    token = 'DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397'

    results, embeddings = parallel_annealing(alpha, beta, token, graphs, chain_strength)
    if results:
        interpreted_results = interpret_combined_results(results, graphs)
        check_results = check_graph_coloring_result(interpreted_results, graphs)
        print(check_results)

if __name__ == "__main__":
    all_results = []
    chain_strength = 2
    
    res = []
    start_time_chain = time.time()
    print(f"Chain strength {chain_strength}")
    for x in range(1, 3):
        print(f"Running test {x}")
        graphs = [
            read_graph_from_file(f"Test/2111/5_{x}.txt"),
            read_graph_from_file(f"Test/2111/10_{x}.txt"),
        ]
        
        # Lưu kết quả của mỗi lần chạy vào all_results
        main_result = main(graphs, chain_strength)
        res.append({
            "test": x,
            "result": main_result
        })

    end_time_chain = time.time()
    elapsed_time_chain = end_time_chain - start_time_chain
    print(f"Total time taken for chain strength {chain_strength}: {elapsed_time_chain:.2f} seconds")
    
    all_results.append({
        "chainstrength": chain_strength,
        "running_time": elapsed_time_chain,
        "result": res
    })

    # Lưu tất cả kết quả vào một file JSON duy nhất
    with open('Res/find_chainstrength/test.txt', 'w') as f:
        json.dump(all_results, f, indent=4)
