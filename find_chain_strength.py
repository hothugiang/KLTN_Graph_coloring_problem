import json
import dimod
import minorminer
import os
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite
import dwave.embedding
import time
import pulp

num_colors = 6

def read_graph_from_file(file_path):
    with open(file_path, "r") as file:
        edge_list = eval(file.read()) 
        graph = nx.Graph(edge_list)
    min_colors = ilp_graph_coloring(graph)
    return graph, min_colors

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

    # H_obj: Tối đa hóa số màu không được sử dụng
    for c in range(num_colors):
        # Thêm biến x_c cho việc màu c không được sử dụng
        bqm.add_variable(f'x_{graph_id}_{c}', -1)

        # Ràng buộc: Nếu có bất kỳ đỉnh nào sử dụng màu c, thì x_c phải bằng 0
        for i in V:
            if f'q_{i}_{graph_id}_{c}' in bqm.variables:
                # Nếu q_{i,c} = 1, thì x_c phải bằng 0 => Thêm ràng buộc tương tác để duy trì điều kiện này
                bqm.add_interaction(f'q_{i}_{graph_id}_{c}', f'x_{graph_id}_{c}', 1)

    return bqm

def find_embeddings(bqm_list, hardware_edges):
    embeddings = []
    available_edges = set(hardware_edges)  # Tập các cạnh khả dụng

    for idx, bqm in enumerate(bqm_list):
        available_graph = nx.Graph()
        available_graph.add_edges_from(available_edges)

        # Lấy danh sách các cạnh từ BQM
        source_edgelist = list(bqm.quadratic.keys())

        start_time_embedding = time.time()
        embedding = minorminer.find_embedding(source_edgelist, available_graph.edges())
        end_time_embedding = time.time()

        embedding_time = end_time_embedding - start_time_embedding

        if embedding:
            embeddings.append((embedding, embedding_time))

            # Cập nhật các cạnh khả dụng sau khi sử dụng
            used_qubits = {qubit for qubits in embedding.values() for qubit in qubits}
            available_edges = {
                edge for edge in available_edges
                if not (edge[0] in used_qubits or edge[1] in used_qubits)
            }
        else:
            print(f"Không thể tìm thấy embedding phù hợp cho bài toán {idx + 1}.")

    return embeddings

def sample_bqm(bqm_combined, bqm_list, graphs, ilp_colors, embeddings, sampler, annealing_time, num_reads=1000):
    start_time = time.time()
    sampleset = sampler.sample(bqm_combined, num_reads=num_reads, annealing_time=annealing_time)
    end_time = time.time()
    total_time = end_time - start_time

    total_solutions = len(sampleset)

    # Thống kê kết quả cho từng bài toán
    stats = {
        graph_id: {
            "success_count": 0,
            "total_constraint_violations": 0,
            "best_node_color_mapping": None,
            "best_num_colors_use": 10,
            "min_constraint_violations": 9999,
            "ilp_num_colors": 0,
        }
        for graph_id in range(len(graphs))
    }

    for sample in sampleset.samples():
        solution = {"samples": dict(sample)}
        interpreted_samples = interpret_and_check_results(solution, embeddings, bqm_list, graphs, ilp_colors)
        
        # Duyệt qua từng bài toán
        for result in interpreted_samples:
            graph_id = result["graph_id"]
            if result["success"] == 1:
                stats[graph_id]["success_count"] += 1

            stats[graph_id]["total_constraint_violations"] += result["constraint_violations"]
            stats[graph_id]["ilp_num_colors"] = result["ilp_num_colors"]

            # Lưu lại kết quả tốt nhất theo thứ tự ưu tiên
            if result["constraint_violations"] == 0:
                if ((stats[graph_id]["min_constraint_violations"] == 0)):
                    if (result["num_colors_used"] < stats[graph_id]["best_num_colors_use"]):
                        stats[graph_id]["best_node_color_mapping"] = result["node_color_mapping"]
                        stats[graph_id]["best_num_colors_use"] = result["num_colors_used"]
                        stats[graph_id]["min_constraint_violations"] = result["constraint_violations"]
                else:
                    stats[graph_id]["best_node_color_mapping"] = result["node_color_mapping"]
                    stats[graph_id]["best_num_colors_use"] = result["num_colors_used"]
                    stats[graph_id]["min_constraint_violations"] = result["constraint_violations"]
            else:
                if (stats[graph_id]["min_constraint_violations"] != 0):
                    if result["constraint_violations"] < stats[graph_id]["min_constraint_violations"]:
                        stats[graph_id]["best_node_color_mapping"] = result["node_color_mapping"]
                        stats[graph_id]["best_num_colors_use"] = result["num_colors_used"]
                        stats[graph_id]["min_constraint_violations"] = result["constraint_violations"]
    
    result = {
        "total_solutions": total_solutions,
        "Time_per_sample": total_time / total_solutions,
        "info": sampleset.info,
        "res": []  # Lưu các kết quả thống kê vào danh sách
    }

    min_P_sol = 1  

    # In thống kê
    for graph_id, stat in stats.items():
        P_sol = stat['success_count'] / total_solutions * 100.000
        res = {
            "graph_id": graph_id,
            "success_count": stat['success_count'],
            "Total_constraint_violations": stat['total_constraint_violations'],
            "Best_node_color_mapping": stat['best_node_color_mapping'],
            "Best_number_of_colors_used": stat['best_num_colors_use'],
            "ILP_num_colors": stat['ilp_num_colors'],
            "Minimum_constraint_violations": stat['min_constraint_violations'],
            "P_sol": P_sol,
        }
        # if (P_sol == 0): res = fix_graph_coloring(res, graphs)
        
        result["res"].append(res)  
        print (result)

    return result

def fix_graph_coloring(res, graphs):
    graph = graphs[res["graph_id"]]

    node_color_mapping = res["Best_node_color_mapping"].copy()
    used_colors = set(node_color_mapping.values())
    num_colors_used = res["Best_number_of_colors_used"]

    # Sửa ràng buộc H1: mỗi đỉnh chỉ được tô duy nhất một màu
    for node in graph.nodes():
        node_colors = [color for n, color in node_color_mapping.items() if n == node]
        if len(node_colors) > 1:
            for color in used_colors:
                # Kiểm tra xem màu này có hợp lệ với đỉnh
                if all(node_color_mapping.get(neighbor) != color for neighbor in graph.neighbors(node)):
                    node_color_mapping[node] = color
                    break
            else:
                # Nếu không có màu hợp lệ, thêm một màu mới
                new_color = max(used_colors) + 1
                node_color_mapping[node] = new_color
                used_colors.add(new_color)
                num_colors_used += 1

    # Sửa ràng buộc H2: không có 2 đỉnh kề nào cùng màu
    for u, v in graph.edges():
        if node_color_mapping[u] == node_color_mapping[v]:
            # Chọn một trong hai đỉnh để sửa màu
            for color in used_colors:
                # Kiểm tra xem màu này có hợp lệ với đỉnh u
                if color != node_color_mapping[v] and all(node_color_mapping.get(neighbor) != color for neighbor in graph.neighbors(u)):
                    node_color_mapping[u] = color
                    break
            else:
                # Nếu không có màu hợp lệ, thêm một màu mới
                new_color = max(used_colors) + 1
                node_color_mapping[u] = new_color
                used_colors.add(new_color)
                num_colors_used += 1
    
    res["Best_node_color_mapping"] = node_color_mapping
    res["Best_number_of_colors_used"] = num_colors_used

    return res
    

def parallel_annealing(alpha, beta, token, graphs, ilp_colors, chain_strength, annealing_time):
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
    result = sample_bqm(bqm_combined, bqm_list, graphs, ilp_colors, [embedding for embedding, _ in embeddings], SimulatedAnnealingSampler(), annealing_time)
    
    return result, [embedding for embedding, _ in embeddings]

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

def interpret_and_check_results(result, embeddings, bqm_list, graphs, ilp_colors):
    interpreted_results = []

    for graph_id, (bqm, embedding) in enumerate(zip(bqm_list, embeddings)):
        node_color_mapping = {}
        used_colors = set()
        graph = graphs[graph_id]
        
        # Duyệt qua từng biến trong BQM và tìm xem biến đó được ánh xạ đến qubit nào
        for variable in bqm.variables:
            if variable.startswith('q_') and variable in embedding:  # Bỏ qua các biến 'color_used' và chỉ xét 'q_'
                qubits = embedding[variable]
                for qubit in qubits:
                    if qubit in result['samples'] and result['samples'][qubit] == 1:
                        parts = variable.split('_')
                        if len(parts) == 4:
                            node_id = int(parts[1])
                            color = int(parts[3])
                            node_color_mapping[node_id] = color
                            used_colors.add(color)

        # Sắp xếp lại thứ tự các nút theo thứ tự tăng dần của node_id
        sorted_node_color_mapping = dict(sorted(node_color_mapping.items()))

        # Kiểm tra kết quả tô màu đồ thị
        valid = True
        constraint_violations = 0

        for node in graph.nodes():
            if node not in sorted_node_color_mapping:
                valid = False
                constraint_violations += 1
            else:
                assigned_colors = [color for n, color in sorted_node_color_mapping.items() if n == node]
                if len(assigned_colors) != 1:
                    valid = False
                    constraint_violations += 1

        if valid:
            for (u, v) in graph.edges():
                if sorted_node_color_mapping.get(u) == sorted_node_color_mapping.get(v):
                    valid = False
                    constraint_violations += 1

        ilp_num_colors = ilp_colors[graph_id]  # Lấy số màu tối thiểu từ mảng `ilp_colors`
        success = 1 if valid and len(used_colors) == ilp_num_colors else 0

        interpreted_results.append({
            "graph_id": graph_id,
            "node_color_mapping": sorted_node_color_mapping,
            "num_colors_used": len(used_colors),
            "ilp_num_colors": ilp_num_colors,
            "valid": valid,
            "constraint_violations": constraint_violations,
            "success": success
        })

    return interpreted_results


def main(graphs, ilp_colors, chain_strength, annealing_time):
    alpha = 1
    beta = 1
    # token = 'DEV-9b90add23224507da2804d959794286a0af0cf8c'
    # token = 'DEV-4779138cb62c490192750836b2d0dcfd2834378c'
    token = 'DEV-0893c79f43c44e4fc6ac83f4c19486308fc62331'

    # Tạo BQM cho từng đồ thị và thực hiện parallel annealing
    bqm_list = [create_bqm_for_graph(graph, alpha, beta, graph_id) for graph_id, graph in enumerate(graphs)]
    results, embeddings = parallel_annealing(alpha, beta, token, graphs, ilp_colors, chain_strength, annealing_time)
        
    return results

if __name__ == "__main__":
    annealing_time = 20
    for N in [5, 10, 15, 20]:
        print(f'Size: ', N)
        all_results = []
        for chain_strength in [1, 2, 3, 5, 7, 9]:
            print(f'chain_strength: ', chain_strength)
            start_time = time.time()
            res = []
            res.append({
                "chain_strength": chain_strength
            })
            for x in range(1, 31):    
                graph1, chroma1 = read_graph_from_file(f"Test/final/{N}_{x}.txt")
                # graph2, chroma2 = read_graph_from_file(f"Test/final/20_{x}.txt")
                graphs = [
                    graph1,
                    # graph2
                ]
                ilp_colors = [
                    chroma1, 
                    # chroma2
                ]
                
                # Lưu kết quả của mỗi lần chạy vào all_results
                results = main(graphs, ilp_colors, chain_strength, annealing_time)
                res.append({
                    "Graph": f'{N}_{x}',
                    "results": results
                })

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Total time taken for chain strength {annealing_time}: {elapsed_time:.2f} seconds")
            
            all_results.append({
                "running_time": elapsed_time,
                "result": res
            })

        # Lưu tất cả kết quả vào một file JSON duy nhất
        with open(f'Res/final/find_chain_strength/n={N}.json', 'w') as f:
            json.dump(all_results, f, indent=4)
