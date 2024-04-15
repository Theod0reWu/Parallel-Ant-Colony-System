import matplotlib.pyplot as plt
import sys

def display(fn):
	data = []
	with open(fn) as f:
		data = f.read().split("\n")
	x, y = [], []
	for i in data:
		point = i.split(",")
		x.append(float(point[0]))
		y.append(float(point[1]))

	plt.scatter(x, y)
	plt.show()

def main():
	display(sys.argv[1])

if __name__ == '__main__':
	main()