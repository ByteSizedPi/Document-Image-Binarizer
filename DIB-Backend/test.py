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

# print(B1)
# im.binarize()

# A = SetOperation(im.img)


# def dilate(): return A + B
# def erode(): return A - B
# def open(): return (A - B) + B
# def close(): return (A + B) - B


# (A - B).save()


dim = 21
B = [[1] * dim] * dim
B1 = [[0] * (dim + 2)] * (dim + 2)
B1[0] = [1] * (dim + 2)
B1[len(B1) - 1] = [1] * (dim + 2)

for row in B1:
    row[0] = row[len(B1) - 1] = 1
