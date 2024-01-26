#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
people = ['Farrah', 'Fred', 'Felicia']

plt.title("Number of Fruit per Person")
# replacing x labels
positions = range(len(people))
plt.xticks(positions, people)
plt.ylabel("Number of fruits")
plt.bar(range(3), fruit[0], color='r', label='apples')
plt.bar(range(3), fruit[1], color='yellow', label='bananas', bottom=fruit[0]+1)
plt.bar(range(3), fruit[2], color='orange', label='oranges', bottom=fruit[0]+fruit[1])
plt.bar(range(3), fruit[3], color='khaki', label='peaches', bottom=fruit[0]+fruit[1]+fruit[2])

plt.legend()
plt.show()
