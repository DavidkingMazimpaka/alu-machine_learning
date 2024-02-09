#/usr/bin/env python3

""" 
    Poison distribution Module which is a discrete probability distribution 
    that describes the probability of a given number of events occurring in a fixed period of time.
    the formula is  f(x) = P(X = x) = \frac{e^{-λ} λ^x}{x!} where x is the number of events, λ is the average number of events, e is the Euler's number and x! is the factorial of x.
"""


class Poisson:
    """Poisson class"""
    def __init__(self, data=None, lambtha=1.):
        """Constructor"""
        self.e = 2.7182818285
        self.pi = 3.1415926536

        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, x):
        """Factorial function"""
        fact = 1
        if x < 0:
            return None
        for i in range(1, x + 1):
            fact *= i
        return fact
    
    def pmf(self, k):
        """Probability mass function"""
        if k < 0:
            return 0
        return ((self.lambtha ** k) * (self.e ** (-self.lambtha))) / self.factorial(k)

    def cdf(self, k):
        """Cumulative distribution function"""
        if k < 0:
            return 0
        return 1 - self.pmf(k)