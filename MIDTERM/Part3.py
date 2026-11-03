import numpy as np

print("=" * 50)
print("PART III – ARRAY OPERATIONS & BROADCASTING")
print("=" * 50)

arr1 = np.array([1, 2, 3, 4])
print("1.", arr1 + 5)

print("2.", np.array([10, 20, 30]) - np.array([1, 2, 3]))

arr3 = np.arange(9).reshape(3, 3)
print("3.\n", arr3 * 10)

p = np.array([1, 2, 3, 4])
q = np.array([1, 0, 3, 0])
print("4.", p == q)

arr5 = np.array([10, 55, 30, 80, 45, 99])
print("5.", arr5[arr5 > 50])

print("6.", np.array([10, 20, 30, 40]) / 5)

print("7.", np.array([1, 2, 3, 4]) ** 2)

arr8 = np.array([-3, 5, -1, 7, -8, 2])
print("8.", np.where(arr8 < 0, 0, arr8))

col = np.array([[1], [2], [3]])
row = np.array([[10, 20, 30]])
print("9.\n", col + row)

arr10 = np.arange(1, 16)
print("10.", arr10 % 3 == 0)