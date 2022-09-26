import time

# iter = 1_000_000_000

# start = time.perf_counter()
# B = [1] * iter
# print(len(B))
# end = time.perf_counter()
# print(end - start)

l = [[1] * 5, [2] * 5, [3] * 5, [4] * 5]
l.sort(key=lambda x: x[4])
print([*l])
