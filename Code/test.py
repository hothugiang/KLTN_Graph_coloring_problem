import dwave_networkx as dnx
import matplotlib.pyplot as plt
from dwave.system.samplers import DWaveSampler

sampler = DWaveSampler(token='DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397')

# Retrieve nodes and edges from the actual hardware graph
hardware_nodes = set(sampler.nodelist)
hardware_edges = set(tuple(sorted(edge)) for edge in sampler.edgelist)

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
h = {0: -1.2,
     1: 0.5,
     2: -0.3,
     3: 1.0,
     4: -0.8,
     5: 0.6,
     6: -1.5,
     7: 0.7,
     8: -0.4,
     9: 1.2}  # Linear biases for qubits 0 and 1
J = {
    (0, 1): -0.6,
    (0, 2): 0.3,
    (1, 3): -0.5,
    (2, 4): -1.1,
    (3, 5): 0.9,
    (4, 6): -0.7,
    (5, 7): 0.8,
    (6, 8): -1.3,
    (7, 9): 0.5,
    (8, 0): -0.4,
    (9, 2): 1.1,
    (3, 8): -0.6,
    (4, 7): 0.9,
    (1, 6): -0.3
}
problem_graph = list(J.keys())

# Find an embedding from the logical problem graph to the hardware graph

from minorminer import find_embedding

embedding = find_embedding(problem_graph, hardware_edges)
used_nodes = []
chains = []
for logical_qubit, physical_qubits in embedding.items():
    used_nodes.extend(physical_qubits)
    chain_edges = [(physical_qubits[i], physical_qubits[i + 1]) for i in range(len(physical_qubits) - 1)]
    chains.extend(chain_edges)

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
    edge_colors.append('lightgray')
# minorminer.find_embedding()
# Redraw with missing nodes and edges highlighted
dnx.draw_pegasus(ideal_graph, node_color=node_colors, edge_color=edge_colors, node_size=5, width=1)

# Add a title for clarity/-strong/-heart:>:o:-((:-h plt.title('Missing Nodes (Red) 
plt.title('Missing Nodes (Red) and Missing Edges (Blue) in D-Wave Hardware Pegasus Graph')
plt.show()