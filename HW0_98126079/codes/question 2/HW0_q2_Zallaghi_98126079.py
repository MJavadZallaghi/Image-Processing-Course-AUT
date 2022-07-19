# DIP Course - fall 2020 - HW: 0
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 2 code

# function definetion
def isPrime(n):
    n = int(n)
    output = False
    counter = 0
    falseMakers = []
    for i in range(1,n+1):
        x = n%i
        if x == 0:
            counter += 1
            falseMakers.append(i)
    if counter == 2:
        output = True
        print(n, " is a prime number!\n\n")
    else:
        print(n, " is not a prime number!\nsee below:\n", falseMakers, ": list of perfect devidable numbers...\n\n")
    return output

# validating isPrime(n) performance
a = isPrime(2)
b = isPrime(4)
c = isPrime(7)
d = isPrime(8)
e = isPrime(11)
f = isPrime(24)

print("Printing bool value returned by isPrime() for upper subjects\n",a,b,c,d,e,f)


        
