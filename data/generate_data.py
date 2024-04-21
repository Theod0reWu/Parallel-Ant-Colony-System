import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import random

import sys
import tkinter

from display_data import display

# Generates points distributed on a disc of radius r, centered at (0,0)
def generate_points_on_disc(num_points, radius = 5):
	points=[]
	for _ in range (num_points):
		theta = random.uniform(0,2*math.pi)
		r = random.uniform(0,radius)
		x=round(r*math.cos(theta),1)
		y=round(r*math.sin(theta),1)
		points.append((x,y))
	return points

def generate_points_on_circle(num_points, radius = 10):
	points = []
	theta_prime = 2 * math.pi / (num_points)
	for i in range(num_points):
		theta = i * theta_prime
		x=round(radius*math.cos(theta),1)
		y=round(radius*math.sin(theta),1)
		points.append((x,y))
	return points

def write_to_file(points,filename):
	with open(filename, 'w') as f:
		for i in range(len(points)):
			if(i != len(points)-1):
				f.write(f"{points[i][0]},{points[i][1]}\n")
			else:
				f.write(f"{points[i][0]},{points[i][1]}")

def main():
	filename = sys.argv[1]
	num_points = int(sys.argv[2])
	distribution = 'circle'
	if (len(sys.argv) == 4):
		distribution = sys.argv[3]

	points = None
	if (distribution == 'circle'):
		points = generate_points_on_circle(num_points)
	elif (distribution == 'disc'):
		points = generate_points_on_disc(num_points)
	write_to_file(points,filename)

if __name__ == '__main__':
	main()