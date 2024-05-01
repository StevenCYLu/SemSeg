import numpy as np

# 90 degree rotations for (d)
A = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
B = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# 180 degree rotations for (e)
D = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
E = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
F = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# Calculate the products
result_d = np.linalg.multi_dot([A, B, C])
result_e = np.linalg.multi_dot([D, E, F])

print("Result for (d):")
print(result_d)
print("Result for (e):")
print(result_e)