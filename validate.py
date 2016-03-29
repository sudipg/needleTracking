#!/usr/bin/python
"""
Script to validate the needle tracking algorithm and evaulate scores 
for hyper paramters on the set of training images
"""

from __future__ import division
from needle_tracking import *
from scipy import misc, ndimage
import argparse
import glob
from matplotlib import pyplot as plt 
import numpy as np

def main():
	#arg parsing 
	parser = argparse.ArgumentParser(description='Detect needles in image set \
		and evaulate probability distribution. generate a score for the given \
		hyper paramters')
	parser.add_argument("-d", "--dir", type=str, help="test directory",default="images/", required=False)
	parser.add_argument("-f", "--fineness", type=int, help="fineness",default=1e4, required=False)
	parser.add_argument("-p", "--probability_threshold", type=float, help="fineness",default=1.5e-2, required=False)
	parser.add_argument("-v", "--verbose", help="debug printing enable", action="store_true")
	args = parser.parse_args()
	test_directory = args.dir
	fineness = args.fineness
	pt = args.probability_threshold
	verbose = args.verbose
	print "The test directory is "+str(test_directory)
	# gather images
	image_filenames = glob.glob(test_directory+'/*.jpeg')
	image_filenames += glob.glob(test_directory+'/*.png')
	image_filenames += glob.glob(test_directory+'/*.jpg')

	print "the image files identified are : "+str(image_filenames)

	scores = dict()

	# sort through the detected images and use the ones with an associated validation file
	for image_filename in image_filenames:
		# look for the corresponding validation file
		image_name = image_filename.split('.')[0]
		validation_filename = image_name+'.needle_data'
		if glob.glob(validation_filename):
			print("working on :"+validation_filename)
			scores[image_filename] = validate_image(image_filename, validation_filename, fineness = fineness, verbose=verbose, probability_threshold = pt)
	print scores
	plt.figure()
	plt.bar(range(len(scores)), [value[0] for value in scores.values()], 0.3,color='b')
	plt.bar(np.array(range(len(scores)))+0.3, [value[1] for value in scores.values()], 0.3,color='r')
	plt.grid()
	plt.show()
	
if __name__ == '__main__':
	main()