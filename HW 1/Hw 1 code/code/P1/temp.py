import numpy as np

def huding(x):
	return x*3

def add(a,b):
	return a(b)

y = huding
print y(3)

print add(y,10)