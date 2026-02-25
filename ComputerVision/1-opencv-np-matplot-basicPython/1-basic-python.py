import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Hello world")

a = 1
b = 1.1
c = "Hello world"
d = True

print(type(d))

# e = int(input("\nHellaw: "))
# print(e)

n = 5
if n == 5:
    print("n is 5")
elif n < 5:
    print("n is less than 5")
else:
    print("n is more than 5")
    
    
if "budi" is "budi":
    print("Bud")
    
    
for i in range(5):
    print(i)
    

i = 0
while i < 5:
    print(i)
    i += 1
    

# List -> support type data yang berbeda
List = [1, 2, 3, "Halo"]
List.append([1, 2, 3])
print(List[4][1])


# Tuples
Tuples = (100, 200)
print(Tuples)


# Set -> unordered, gapunya urutan; kalo list gabisa duplicate, bakal ilang
Set = {1, 2, 3, "Halo", "Dunia"}
# print(Set[0])


# Dictionary -> yang di depan adalah index-nya
Dictionary = {
    "Halo": 1,
    2: "Dunia"
}
print(Dictionary["Halo"])


# Slicing
N = [0, 10, 20, 30, 40, 50, 60]
print(N[1:4])
print(N[::-1])