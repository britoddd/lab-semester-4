import numpy as np

# Basic Array
arr = np.array([[1, 2], [3, 4]])
print(arr)
print(arr.shape) # Ukuran array-nya (berapa kali berapa)
print(arr.ndim)


# Array of Ones (putih)
ones = np.ones((1,2))  # Shape-nya, input parameternya kudu tuple
print(ones)

# Array of Zeros (item)
zeros = np.zeros((3,2))
print(zeros)

# Empty Array (kosongan, isinya garbage value)
empty = np.empty((2,2))
print(empty)

# Array of Randoms
random = np.random.rand(2,2)  # khusus ini ngga pake tuple
print(random)

random = np.random.randn(2,2)  # random normal
print(random)

random = np.random.randint(0, 255, (10,10))
print(random)


# Stack
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Hstack => yah, take it literally lah, bakal jejer ke samping
c = np.hstack((a, b))
print(c)

# Vstack => kalo yang ini bakal jadi jejer ke bawah
d = np.vstack((a, b))
print(d)