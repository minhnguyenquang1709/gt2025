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

    print(f"Directed matrix of g:\n{g.directed_matrix}")
    print(f"Number of directed edges: {len(g.edges)}")
    print(f"Undirected matrix of g:\n{g.undirected_matrix}")
