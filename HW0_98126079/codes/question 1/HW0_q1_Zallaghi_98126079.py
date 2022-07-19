# DIP Course - fall 2020 - HW: 0
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 1 code

# using from scipy integration module
import scipy.integrate as integrate
# using numpy for making array
import numpy as np
# importing sin function
from math import sin

# defining of function below inegartion
def sin2x (x):
    return sin(x*x)

# making integration using simpson function
a = -3
b = 4
step = 1/200
num = int((b-a)/step)
x_range = np.linspace(a, b, num)
y_range =  np.array([sin2x(xi) for xi in x_range])
resultSimpson = integrate.simps(y_range, x_range, step)

# making integration using quad function
resultQuad = integrate.quad(sin2x, a, b)

# printing results
print("Integration with fixed step size - Simpson approach: ", resultSimpson,
      "\nIntegration with variable step size - Quad approach: ", resultQuad)

