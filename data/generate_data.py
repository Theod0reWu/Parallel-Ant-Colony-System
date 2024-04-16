import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import random

import sys
import tkinter

from display_data import display

def generate_points(num_points):
	points=[]
	radius = 5
	for _ in range (num_points):
		theta = random.uniform(0,2*math.pi)
		r = random.uniform(0,radius)
		x=round(r*math.cos(theta),1)
		y=round(r*math.sin(theta),1)
		points.append((x,y))
	print(points)
	return points

def write_to_file(points,filename):
	with open(filename, 'w') as f:
		for i in range(len(points)):
			if(i != len(points)-1):
				f.write(f"{points[i][0]},{points[i][1]}\n")
			else:
				f.write(f"{points[i][0]},{points[i][1]}")
			


def main():
	filename = "data.txt"
	points = generate_points(10)
	write_to_file(points,filename)
	# display(sys.argv[1])

if __name__ == '__main__':
	main()