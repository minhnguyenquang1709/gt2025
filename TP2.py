from graphs import *

if __name__ == "__main__":
    vertex_names = "123456789"

    g = DirectedGraph()

    g.add_vertices_by_string(vertex_names)

    edge_names = [
        (1, 2),
        (1, 4),
        (2, 3),
        (2, 6),
        (5, 4),
        (5, 5),
        (5, 9),
        (6, 3),
        (6, 4),
        (7, 3),
        (7, 5),
        (7, 6),
        (7, 8),
        (8, 3),
        (8, 9),
    ]

    g.add_directed_edges(edge_names)

    print(f"Degree of Vertex(3) is: {g.degree(3)}\n")
    print(f"Degree of Vertex(4) is: {g.degree('4')}\n")

    print(f"Directed matrix of g:\n{g.show_directed_matrix()}")
    print(f"Number of directed edges: {len(g.edges)}")
    print(f"Undirected matrix of g:\n{g.show_undirected_matrix()}")

    print(f"BFS of g from Vertex(1) is:\n{g.bfs_directed(1)}\n")
    print(f"DFS of g from Vertex(1) is:\n{g.dfs_directed(1)}\n")

    print(f"BFS result for the underlying undirected graph of g from Vertex(1) is:\n{g.bfs(1)}\n")
    print(f"DFS result for the underlying undirected graph of g from Vertex(1) is:\n{g.dfs(1)}\n")

    # find components within graph
    g.find_components()
    g.find_scc()
    print(f"Weakly connected components in g:\n{g.components}")
    for component in g.components:
        print(f"---WCC vertices:{component.vertices}")
        print(f"---WCC edges:{component.edges}")
        print(f"---WCC's undirected matrix:\n{component.show_undirected_matrix()}\n")
        print(f"---g's directed matrix:\n{g.show_directed_matrix()}\n")
        print(f"---WCC's directed matrix:\n{component.show_directed_matrix()}\n")

    print(f"Strongly connected components in g:\n{g.scc}")
    for component in g.scc:
        print('--------------------------------------------')
        print(f"---SCC vertices:{component.vertices}")
        print(f"---SCC edges:{component.edges}")
        print(f"---SCC's undirected matrix:\n{component.show_undirected_matrix()}\n")
        print(f"---SCC's directed matrix:\n{component.show_directed_matrix()}\n")
