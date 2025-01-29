from graphs import *

g = Graph()

vertices = "ABCDEFGHLM"
g.add_vertices_by_string(vertices)

edges = [
    ('A', 'C', 1),
    ('A', 'B', 4),
    ('B', 'F', 3),
    ('C', 'D', 8),
    ('C', 'F', 7),
    ('D', 'H', 5),
    ('F', 'E', 1),
    ('F', 'H', 1),
    ('E', 'H', 2),
    ('E', 'L', 2),
    ('H', 'G', 3),
    ('H', 'M', 7),
    ('H', 'L', 6),
    ('G', 'L', 4),
    ('G', 'M', 4),
    ('L', 'M', 1)
]
g.add_undirected_edges(edges)

# print(f"Undirected matrix:\n{g.show_umatrix()}")

print("Adjacency Matrix:")
print(g.show_wmatrix())

# Get user input
source = input("Enter the source vertex (S): ")
target = input("Enter the target vertex (T): ")

try:
    path, total_weight = g.dijkstra(source, target)
    if path:
        print(f"Shortest path from {source} to {target}: {' -> '.join(path)}")
        print(f"Total weight: {total_weight}")
    else:
        print(f"No path exists from {source} to {target}.")
except ValueError as e:
    print(e)