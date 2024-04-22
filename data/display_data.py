import matplotlib.pyplot as plt
import sys

def get_data(fn):
	data = []
	with open(fn) as f:
		data = f.read().split("\n")
	x, y = [], []
	for i in data:
		point = i.split(",")
		x.append(float(point[0]))
		y.append(float(point[1]))
	return x, y

def display_solution(fn):
	fig, ax = plt.subplots()
	x, y = get_data(fn)
	plt.scatter(x, y)
	xl = x + [x[0]]
	yl = y + [y[0]]
	plt.plot(xl, yl, linewidth = 1, linestyle='dotted', color = 'b')

	total = 0
	for i in range(len(x)):
		total += ((xl[i] - xl[i+1]) ** 2 + (yl[i] - yl[i+1]) ** 2) ** .5
	print("total score:", total)

	for i, txt in enumerate([i for i in range(len(x))]):
		ax.annotate(txt, (x[i], y[i]))

	plt.show()

def display(fn):
	x, y = get_data(fn)
	plt.scatter(x, y)
	plt.show()

def main():
	solution = False
	if (len(sys.argv) == 2):
		solution = True
	if (solution):
		display(sys.argv[1])
	else:
		display_solution(sys.argv[1])

if __name__ == '__main__':
	main()