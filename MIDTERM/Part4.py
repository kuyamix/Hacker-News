import numpy as np

print("=" * 50)
print("PART IV – STATISTICS & AGGREGATION")
print("=" * 50)

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
print("1.", np.sum(arr1))

arr2 = np.array([10, 20, 30, 40, 50])
print("2. mean:", np.mean(arr2), "| std:", np.std(arr2))

arr3 = np.random.randint(1, 100, 10)
print("3. max:", np.max(arr3), "| min:", np.min(arr3))

arr4 = np.array([[1, 2, 3], [4, 5, 6]])
print("4.", np.mean(arr4, axis=1))

arr5 = np.array([[1, 2, 3], [4, 5, 6]])
print("5.", np.sum(arr5, axis=0))

arr6 = np.array([1, 2, 3, 4, 5])
print("6.", np.prod(arr6))

arr7 = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("7.", np.median(arr7))

arr8 = np.array([50, 120, 80, 150, 200, 30])
print("8.", np.sum(arr8 > 100))

arr9 = np.random.randint(1, 100, 20)
print("9. mean:", np.mean(arr9))

arr10 = np.array([10, 20, 30, 40, 50])
print("10.", np.var(arr10))