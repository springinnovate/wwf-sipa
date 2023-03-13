import matplotlib.pyplot as plt
import numpy

# initializing the data
x = numpy.linspace(0, 1)

y = numpy.cos(x*numpy.pi)/2+0.5

# plotting the data
plt.plot(x, y)

# Adding the title
plt.title("Simple Plot")

# Adding the labels
plt.ylabel("y-axis")
plt.xlabel("x-axis")
plt.show()
