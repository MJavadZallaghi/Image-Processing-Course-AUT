# DIP Course - fall 2020 - HW: 0
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 4 code

# importing modules
import numpy as np
from matplotlib import pyplot as plt

# defining function for making derivatives
def polDerCoefVec(coeffVec):
  rank = np.size(coeffVec) - 1
  outPut = np.zeros(rank)
  for i in range(rank):
    outPut[i] = (rank-i) * coeffVec[i]
  return outPut

# our test polynomial ans its derivative
polCoeffs = np.array([1,2,-5])
polDerivaticeCoeffs = polDerCoefVec(polCoeffs)

# definening range for value of polynomials calculation
x_domain = np.linspace(0,10,500)
y_domain = np.polyval(polCoeffs, x_domain)
ypeime_domain = np.polyval(polDerivaticeCoeffs, x_domain)

# ploting results
plt.plot(x_domain,y_domain, label = "main polynomial", linewidth = 2, color = "blue")
plt.plot(x_domain,ypeime_domain, label = "derivative polynomial", linewidth = 1, ls = '--', color = "red")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.title("Question 4 plot: a polynamial and its derivative")
plt.show()




        
