from graphs import *

g = DGraph()

vertex_names = "123456789"

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

g.add_vertices_by_string(vertex_names)
g.add_directed_edges(edge_names)

print(f"Undirected matrix: \n{g.show_umatrix()}")
print(f"Directed matrix: \n{g.show_dmatrix()}")

wccs, wcc_numb = g.wcc()
print(f"There are {wcc_numb} weakly connected component(s) in graph G")
for component in wccs:
    print(component)

sccs, scc_numb = g.scc()
print(f"There are {scc_numb} strongly connected component(s) in graph G")
for component in sccs:
    print(component)