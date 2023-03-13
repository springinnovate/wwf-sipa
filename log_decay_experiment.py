import matplotlib.pyplot as plt
import numpy

# initializing the data
x = numpy.linspace(0, 1)

y = numpy.sin(numpy.pi+(x**5*numpy.pi))/2+0.5
plt.plot(x, y)
y = numpy.exp(-x)**2
plt.plot(x, y)
# plotting the data

# Adding the title
plt.title("Simple Plot")

# Adding the labels
plt.ylabel("y-axis")
plt.xlabel("x-axis")
plt.show()
