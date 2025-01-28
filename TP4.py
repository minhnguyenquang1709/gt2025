from graphs import *

vertex_names = "123456789"

g = Graph()

g.add_vertices_by_string(vertex_names)

edge_names = [
    (1, 2, 4),
    (1, 5, 1),
    (1, 7, 2),
    (2, 3, 7),
    (2, 6, 5),
    (3, 4, 1),
    (3, 6, 8),
    (4, 6, 6),
    (4, 7, 4),
    (4, 8, 3),
    (5, 6, 9),
    (5, 7, 10),
    (6, 9, 2),
    (7, 7, 2),
    (7, 9, 8),
    (8, 9),
]

g.add_undirected_edges(edge_names)
print(g._edges)

print(f"The weighted matrix representation of the graph is:\n{g.show_wmatrix()}")
print(f"w(1, 9) = {g.w((1, '9'))}")
root = str(7)

v1, e1 = g.prim(root)
print("MST by Prim's algo:")
for e in e1:
    print(e)

print("")

mst = g.kruskal()
print("MST by Kruskal's algo:")
for e in mst:
    print(e)