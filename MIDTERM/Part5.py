import numpy as np

print("=" * 50)
print("PART V – RANDOM & SIMULATION")
print("=" * 50)

print("1.\n", np.random.randint(1, 51, size=(3, 3)))

print("2.", np.random.randint(1, 7, 10))

print("3.", np.random.choice(['Heads', 'Tails'], 20))

print("4.", np.random.rand(5))

r5 = np.random.rand(4, 4)
print("5. sum:", r5.sum())

np.random.seed(42)
first = np.random.rand(5)
np.random.seed(42)
second = np.random.rand(5)
print("6. first :", first)
print("   second:", second)
print("   same? ", np.array_equal(first, second))

print("7.", np.random.choice([10, 20, 30, 40, 50], 3, replace=False))

a8 = np.array([1, 2, 3, 4, 5])
np.random.shuffle(a8)
print("8.", a8)

a9 = np.random.randint(1, 100, 100)
print("9. even count:", np.sum(a9 % 2 == 0))

rolls = np.random.randint(1, 7, 1000)
values, counts = np.unique(rolls, return_counts=True)
print("10.")
for v, c in zip(values, counts):
    print(f"    Face {v}: {c}")