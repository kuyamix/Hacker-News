import numpy as np

print("=" * 50)
print("PART I – BASIC OPERATIONS")
print("=" * 50)

# 1. Create a 1D NumPy array containing numbers 1 to 10
arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("1.", arr1)

# 2. Display the dtype and shape of the array [1, 2, 3, 4, 5]
arr2 = np.array([1, 2, 3, 4, 5])
print("2. dtype:", arr2.dtype, "| shape:", arr2.shape)

# 3. Generate an array of 10 zeros and replace the 5th element with 99
arr3 = np.zeros(10)
arr3[4] = 99          # index 4 = 5th element
print("3.", arr3)

# 4. Create a NumPy array using np.arange(2, 21, 2)
arr4 = np.arange(2, 21, 2)
print("4.", arr4)

# 5. Display only the even elements from [1, 2, 3, 4, 5, 6, 7, 8]
arr5 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print("5.", arr5[arr5 % 2 == 0])

# 6. Create a 3x3 identity matrix
arr6 = np.eye(3)
print("6.\n", arr6)

# 7. Convert [10, 20, 30] into a NumPy array and multiply each by 2
arr7 = np.array([10, 20, 30]) * 2
print("7.", arr7)

# 8. Print the shape and size of an array made from np.arange(12)
arr8 = np.arange(12)
print("8. shape:", arr8.shape, "| size:", arr8.size)

# 9. Use linspace to create 5 equally spaced numbers between 0 and 100
arr9 = np.linspace(0, 100, 5)
print("9.", arr9)

# 10. Create an array and display its last element using negative indexing
arr10 = np.array([10, 20, 30, 40, 50])
print("10.", arr10[-1])