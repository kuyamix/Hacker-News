import numpy as np

print("=" * 50)
print("PART II – ARRAY MANIPULATION")
print("=" * 50)

arr1 = np.arange(12).reshape(3, 4)
print("1.\n", arr1)

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("2.", arr2.ravel())

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
print("3.\n", np.vstack((x, y)))

print("4.\n", np.hstack((x, y)))

arr5 = np.arange(12)
print("5.", np.split(arr5, 3))

print("6.", np.concatenate((np.array([1, 2, 3]), np.array([4, 5, 6]))))

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("7.", m[:, 1])

m8 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("8.\n", np.insert(m8, 1, [9, 9, 9], axis=0))

m9 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("9.\n", np.delete(m9, 1, axis=0))

arr10 = np.arange(10).reshape(2, 5).reshape(5, 2)
print("10.\n", arr10)