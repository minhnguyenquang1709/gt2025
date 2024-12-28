from graphs import *

if __name__ == "__main__":
    vertex_names = "123456789"

    graph = Graph()

    for vertex_name in vertex_names:
        graph.add_vertex(vertex_name)

    edge_names = [
        "12",
        "14",
        "23",
        "26",
        "54",
        "55",
        "59",
        "63",
        "64",
        "73",
        "75",
        "76",
        "78",
        "83",
        "89",
    ]

    graph.add_edges(edge_names)

    print('Vertices: ', graph.get_vertices())
    print('Edges: ', graph.get_edges())
    graph.make_matrix()
    print(f"The matrix representation of the graph is:\n{graph.matrix_rep}")