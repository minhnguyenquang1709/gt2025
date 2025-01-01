from graphs import *

if __name__ == "__main__":
    vertex_names = "123456789"

    graph = Graph()

    graph.add_vertices_by_string(vertex_names)

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

    graph.add_edges(edge_names)

    print("Vertices: ", graph.get_vertices())
    print("Edges: ", graph.get_edges())
    graph.make_matrix()
    print(f"The matrix representation of the graph is:\n{graph.show_matrix()}")

    print(f'Path existence (1, 3): {graph.path_existence(1, 3)}')

    print(f'Path existence (2, 9): {graph.path_existence(2, 9)}')

    print(f'Path existence (3, 5): {graph.path_existence(3, 5)}')

    print(f'Path existence (1, 1): {graph.path_existence(1, 1)}')

    print(f'Path existence (5, 5): {graph.path_existence(5, 5)}')

    print(f'Path existence (10, 3): {graph.path_existence(10, 3)}')

    print(f'Path existence (7, 9): {graph.path_existence(7, 9)}')

    print(f'Path existence (4, 7): {graph.path_existence(4, 7)}')

    print(f'Path existence ("2", "3"): {graph.path_existence("2", "3")}')

    print(f'Path existence ("7", "9"): {graph.path_existence("7", "9")}')