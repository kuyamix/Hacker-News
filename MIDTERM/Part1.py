import numpy as np

print("=" * 50)
print("PART I – BASIC OPERATIONS")
print("=" * 50)

arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("1.", arr1)

arr2 = np.array([1, 2, 3, 4, 5])
print("2. dtype:", arr2.dtype, "| shape:", arr2.shape)

arr3 = np.zeros(10)
arr3[4] = 99
print("3.", arr3)

arr4 = np.arange(2, 21, 2)
print("4.", arr4)

arr5 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print("5.", arr5[arr5 % 2 == 0])

arr6 = np.eye(3)
print("6.\n", arr6)

arr7 = np.array([10, 20, 30]) * 2
print("7.", arr7)

arr8 = np.arange(12)
print("8. shape:", arr8.shape, "| size:", arr8.size)

arr9 = np.linspace(0, 100, 5)
print("9.", arr9)

arr10 = np.array([10, 20, 30, 40, 50])
print("10.", arr10[-1])