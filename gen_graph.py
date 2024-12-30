import networkx as nx

n = [5, 10,15, 20]
c = 4.5

tmp = 0
for i in n:
    p = 0.5
    for j in range(30):
        # tmp = tmp+1
        graph = nx.erdos_renyi_graph(i, p)
        file_name = f"Test/final2/{i}_{j+1}.txt"

        print(graph)

        edge_list = list(graph.edges())

        with open(file_name, "w") as file:
            file.write(str(edge_list))
