from graphs import *

g = DGraph()

vertex_names = "12345678"

edge_names = [(1, 2), (1, 3), (2, 5), (2, 6), (3, 4), (4, 8), (5, 7)]

g.add_vertices_by_string(vertex_names)
g.add_directed_edges(edge_names)

# print(f"Undirected matrix: \n{g.show_umatrix()}")
print(f"Adjacent matrix of G: \n{g.show_dmatrix()}")
# for vertex_name in vertex_names:
#     print(f"In-degree of vertex {vertex_name} = {g.idegree(vertex_name)}")
#     print(f"Out-degree of vertex {vertex_name} = {g.odegree(vertex_name)}\n")

print(f"Traverse the tree G in-order:\n")
g.inorder_traversal(1)