import matplotlib.pyplot as plt
import numpy

# initializing the data
x = numpy.linspace(0, 1)
xp=1
print((numpy.cos((xp*numpy.pi))+1)/2)

y = -(numpy.sin(((x-0.5)*numpy.pi))-1)/2
plt.plot(x, y)
#y = numpy.exp(-x-1)
alpha = 3

def func(x):
    return [
        x[0]*numpy.exp(-alpha*(0-1))+x[1]-1,
        x[0]*numpy.exp(-alpha*(1-1))+x[1]]

from scipy.optimize import fsolve
A, B = fsolve(func, [1, 1])
y = A*numpy.exp(-alpha*(x-1))+B
plt.plot(x, y)

y = 1-x
plt.plot(x, y)
# plotting the data

# Adding the title
plt.title("Simple Plot")

# Adding the labels
plt.ylabel("y-axis")
plt.xlabel("x-axis")
plt.show()
