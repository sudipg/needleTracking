#!/usr/bin/python
"""
Needle tracking
"""
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt 
import cv2
import pdb
from Tkinter import *
import tkFileDialog
import argparse
import random

from color_classifier import is_metallic

verbose = False
zones = None
image = None
img_width = 0
img_height = 0


def main():
	"""
	Main function for needle tracker
	"""


	# arg parsing
	parser = argparse.ArgumentParser(description='Detect needle in images \
		and plot probability distribution.')
	parser.add_argument("-i", "--image", type=str, help="input image filename", default=None, required=False)
	parser.add_argument("-v", "--verbose", help="debug printing enable",
                    action="store_true")
	args = parser.parse_args()
	global verbose 
	verbose = args.verbose
	image = None
	filename = None

	# check if file specified
	if args.image:
		filename = args.image
		try:
			image = misc.imread(filename)
			if verbose:
				print filename
		except:
			print 'something wrong with the filename'
	else:
		filename = 'images/left1008.jpeg'
		image = misc.imread(filename)

	if verbose:
		plt.imshow(image)
		plt.show()

	img_width = len(image[0])
	img_height = len(image)

	num_windows = 1e4 # The number of zones of probability
	global zones 
	zones = np.zeros(((int)np.sqrt(num_windows),(int)np.sqrt(num_windows))) # set up the zones of distinct probability
	set_random_p()


def 



def set_random_p():
	global zones;
	for i in zones.shape[0]:
		for j in zones.shape[1]:
			zones[i,j] = random.random()

if __name__ == '__main__':
	main()