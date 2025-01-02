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

    print("Vertices: ", g.get_vertices())
    print("Edges: ", g.get_edges())
    print(f"The matrix representation of the graph is:\n{g.show_directed_matrix()}")

    print(f"Path existence (1, 3): {g.path_existence(1, 3)}")

    print(f"Path existence (2, 9): {g.path_existence(2, 9)}")

    print(f"Path existence (3, 5): {g.path_existence(3, 5)}")

    print(f"Path existence (1, 1): {g.path_existence(1, 1)}")

    print(f"Path existence (5, 5): {g.path_existence(5, 5)}")

    print(f"Path existence (10, 3): {g.path_existence(10, 3)}")

    print(f"Path existence (7, 9): {g.path_existence(7, 9)}")

    print(f"Path existence (4, 7): {g.path_existence(4, 7)}")

    print(f'Path existence ("2", "3"): {g.path_existence("2", "3")}')

    print(f'Path existence ("7", "9"): {g.path_existence("7", "9")}')

    print("For the Vertex(3)\n")
    print(f"Total degrees: {g.degree(3)}")
    print(f"In-degree: {g.idegree('3')}")
    print(f'Out-degree: {g.odegree("3")}')
