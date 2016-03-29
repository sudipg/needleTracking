#!/usr/bin/python
"""
Needle tracking
"""
from __future__ import division
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt 
import pdb
from Tkinter import *
import tkFileDialog
import argparse
import random
from skimage import measure, filters, data, feature
from mayavi import mlab
from color_classifier import is_metallic, is_metallic_fast
import pygame
import time
from sklearn import decomposition
import b2ac.preprocess
import b2ac.fit
import b2ac.conversion
from skimage.draw import ellipse_perimeter
import copy

verbose = False
zones = None
image = None
image_BW = None
img_width = 0
img_height = 0
p_map = None
num_windows = 0
filename = None
global components


def validate_image(image_filename, validation_filename, fineness=1e4, verbose=False, probability_threshold = 0.1):
	"""
	Take an image filename as an input an return the score of validation procedure
	"""
	global image
	global img_height
	global img_width
	global num_windows
	global filename
	global image_BW
	global zones
	global p_map

	#pdb.set_trace()

	num_windows = fineness
	image = misc.imread(image_filename)
	image_BW = data.imread(image_filename, as_grey=True)
	# clean the existing pmap
	img_width = len(image[0])
	img_height = len(image)
	zones = np.zeros(((int)(np.sqrt(num_windows)),(int)(np.sqrt(num_windows))))
	generate_uniform_p()
	add_metal_segmentation_p()
	add_edge_segmentation_p()

	if(verbose):
		plot_pixel_map();

	# read the validation file and build validation matrix
	# val_coords holds a list of all the x, y coords of points labeled as a needle
	v_file = open(validation_filename)
	val_coords = filter(lambda x: x!='', v_file.read().split('\n'))
	val_map = np.zeros((np.sqrt(fineness), np.sqrt(fineness)))
	v_file.close()
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)

	pixel_val_map = np.zeros((img_height, img_width))

	for coord in val_coords:
		coord = coord.split(',')
		x = int(coord[0])
		y = int(coord[1])
		pixel_val_map[y,x] = 1

	pixel_val_map = filters.gaussian_filter(pixel_val_map, 0.5)
	#pixel_val_map = pixel_val_map > pixel_val_map.max()*0.2

	for x in range(img_width):
		for y in range(img_height):
			if(pixel_val_map[y,x]):
				val_map[y//tmp2,x//tmp1] += 1
	val_map = (val_map > 0).astype('float')
	needle_locs = (zones > zones.mean()*probability_threshold)
	if(verbose):
		plt.clf()
		plt.subplot('311')
		plt.imshow(needle_locs, cmap='gray')
		plt.subplot('312')
		plt.imshow(val_map, cmap='gray')
		plt.subplot('313')
		plt.imshow(image)
		plt.show()
	return (val_map*needle_locs.astype('int')).sum()/val_map.sum(), (needle_locs.sum() - (val_map*needle_locs.astype('int')).sum())/val_map.sum()






def main():
	"""
	Main function for needle tracker
	"""
	# arg parsing
	parser = argparse.ArgumentParser(description='Detect needle in images \
		and plot probability distribution.')
	parser.add_argument("-i", "--image", type=str, help="input image filename", default=None, required=False)
	parser.add_argument("-f", "--fineness", type=float, help="number of windows to split into (must be square)", default=1e4, required=False)
	parser.add_argument("-v", "--verbose", help="debug printing enable",
					action="store_true")
	parser.add_argument("-s", "--shape", help="enable ellipse fitting probability calculation",
					action="store_true")
	args = parser.parse_args()
	global verbose 
	global image
	global img_height
	global img_width
	global num_windows
	global filename
	global image_BW
	verbose = args.verbose
	image = None
	filename = None

	# check if file specified
	if args.image:
		filename = args.image
		try:
			image = misc.imread(filename)
			image_BW = data.imread(filename, as_grey=True)
			if verbose:
				print filename
		except:
			print 'something wrong with the filename'
	else:
		filename = 'images/left1008.jpeg'
		image = misc.imread(filename)
		image_BW = data.imread(filename, as_grey=True)

	if verbose:
		plt.imshow(image)
		plt.show()

	img_width = len(image[0])
	img_height = len(image)

	
	if args.fineness:
		num_windows = args.fineness
	else:
		num_windows = 1e4 # The number of zones of probability must be a perfect square
	global zones 
	zones = np.zeros(((int)(np.sqrt(num_windows)),(int)(np.sqrt(num_windows)))) # set up the zones of distinct probability
	# start off with a uniform probability distribution
	generate_uniform_p()
	print "starting metal segmentation"
	t0 = time.time()
	add_metal_segmentation_p()
	t1 = time.time()
	print "done metal segmentation in "+str(t1-t0)+" s"
	plot_pixel_map()
	print "starting edges segmentation"
	t2 = time.time()
	add_edge_segmentation_p()
	t3 = time.time()
	print "done edge segmentation in "+str(t3-t2)+" s"
	plot_pixel_map()
	if args.shape:
		print "starting ellipse fitting segmentation"
		add_ellipse_segmentation_p()
		t4 = time.time()
		print "done ellipse fitting segmentation in "+str(t4-t3)+" s"
	#plot_pixel_map()
	if verbose:
		plot_components()


def add_metal_segmentation_p():
	global zones
	global image
	global img_height
	global img_width

	metal_zones = np.zeros((zones.shape[0], zones.shape[1]))
	xs = np.r_[0:img_width:1]
	ys = np.r_[0:img_height:1]
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)
	metal_mask = is_metallic_fast(image, new_mean = 1)
	for x in xs:
		for y in ys:
			if metal_mask[y,x]:
				metal_zones[y//tmp2,x//tmp1] += 1
	metal_zones = metal_zones/metal_zones.sum()
	zones = zones*metal_zones
	normalize_map()

def add_edge_segmentation_p():
	global zones
	global image
	global img_height
	global img_width
	global filename
	global components
	global image_BW

		# trying to sharpen the image 
	# edge_max = filters.sobel(image_BW).max()
	# img_blurred = filters.gaussian_filter(image_BW, sigma=2)
	edges = feature.canny(image_BW)
	# # try to kill smaller components
	# labelled_img = measure.label(edges, connectivity=2)

	# num_components = labelled_img.max()
	# #print num_components
	# components = []
	# for i in range(num_components):
	# 	component = (labelled_img == i).astype('int')
	# 	size =  np.count_nonzero(component)
	# 	components.append((component,size))
	# components = sorted(components, key=lambda x: -x[1])
	# # now lets take the biggest max(4, len(components)) components
	# num_components = min(10, len(components))
	# components = components[1:num_components]

	# components = [components[i][0] for i in range(len(components))]
	# components_sum = sum(components)

	# if verbose:
	# 	plt.imshow(components_sum)
	# 	plt.show()

	edge_zones = np.zeros((zones.shape[0], zones.shape[1]))
	xs = np.r_[0:img_width:1]
	ys = np.r_[0:img_height:1]
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)

	for x in xs:
		for y in ys:
			if edges[y,x]:
				edge_zones[y//tmp2,x//tmp1] += 1
	edge_zones = edge_zones**2/edge_zones.sum()
	zones = zones*edge_zones
	normalize_map()

def add_ellipse_segmentation_p():
	global components
	global zones
	global image
	global img_height
	global img_width

	acceptable_A = 300
	acceptable_B = 150

	ellipse_zones = np.zeros((zones.shape[0], zones.shape[1]))


	for target in range(len(components)):
		try:
			y,x = np.nonzero(components[target])
			x = x[::5]
			y = y[::5]
			if verbose:
				plt.imshow(components[target], cmap='gray')
				plt.title('Sampling the component')
				plt.show()
			data = np.array([[x[i], y[i]] for i in range(len(x))])
			pca = decomposition.PCA(n_components=2)
			pca.fit(data)
			data2 = pca.transform(data)
			if verbose:
				plt.plot(data2[:,0],data2[:,1],'r+')
			data2 = list(data2)
			data2.sort(key=lambda x:x[0])
			data2 = np.array(data2)
			if verbose:
				plt.plot(data2[:,0],data2[:,1],'r+')

			# get average pcaY for each window
			num_groups = 10
			size_of_group = len(data2)//num_groups
			if verbose:
				print len(data2)
				print size_of_group
			data2 = data2[:num_groups*size_of_group]
			if verbose:
				print len(data2)
			groups = np.split(data2, num_groups)
			pcaY = []
			pcaX = []

			for group in groups:
				if group.size == 0:
					continue
				pcaY.append(np.mean(group[:,1]))
				pcaX.append(np.mean(group[:,0]))
			if verbose:
				plt.plot(pcaX, pcaY, 'bo')
				plt.show()
			pcaX = np.array(pcaX)
			pcaY = np.array(pcaY)
			pcaData = np.vstack((pcaX, pcaY)).transpose()
			invPcaData = pca.inverse_transform(pcaData)
			points, x_mean, y_mean = b2ac.preprocess.remove_mean_values(invPcaData)
			points = np.array(points)
			conic_numpy = b2ac.fit.fit_improved_B2AC_double(points)
			general_form_numpy = b2ac.conversion.conic_to_general_1(conic_numpy)
			general_form_numpy[0][0] += x_mean
			general_form_numpy[0][1] += y_mean
			A = general_form_numpy[1][1]
			B = general_form_numpy[1][0]

			dist_from_acceptable = np.linalg.norm(np.array([A,B]) - np.array([acceptable_A,acceptable_B]))
			weight = 1/dist_from_acceptable

			xs = np.r_[0:img_width:1]
			ys = np.r_[0:img_height:1]
			tmp1 = img_width/np.sqrt(num_windows)
			tmp2 = img_height/np.sqrt(num_windows)
			for x in xs:
				for y in ys:
					if components[target][y,x]:
						ellipse_zones[y//tmp2,x//tmp1] += weight**2


			if verbose:
				print general_form_numpy
				plt.imshow(components[target], cmap='gray')
				plt.plot(cx,cy, 'r+')
				plt.plot(invPcaData[:,0],invPcaData[:,1],'bo')
				plt.show()
		except Exception:
			continue
	ellipse_zones = ellipse_zones/ellipse_zones.sum()
	zones = zones*ellipse_zones
	normalize_map()


############################
#### HELPER FUNCTIONS ######
############################

def plot_pixel_map():
	global zones
	global image 
	global img_height
	global img_width
	generate_pixel_map()
	plt.imshow(image)
	plt.imshow(p_map, cmap='jet',interpolation='nearest', alpha=0.5)
	plt.show()
	X = range(img_width)
	Y = range(img_height)
	m = mlab.surf(X,Y,p_map, warp_scale='auto', opacity=1)
	mlab.colorbar(m, orientation='vertical')
	mlab.axes(m)
	mlab.xlabel('Y')
	mlab.ylabel('X')
	mlab.zlabel('probability')
	mlab.outline(m)
	mlab.show()
	
	
	

def plot_components():
	global components

	for c in components:
		plt.imshow(c, cmap='gray')
		plt.show()

def generate_pixel_map():
	"""
	generates a map of probabilities that can be overlayed on the image
	"""
	global p_map
	global zones
	p_map = np.zeros((img_height, img_width))
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)
	for i in range(zones.shape[0]):
		for j in range(zones.shape[1]):
			p_map[i*tmp2:(i+1)*tmp2,j*tmp1:(j+1)*tmp1] = zones[i,j]

def get_pixel_zone(x, y):
	global num_windows
	global img_width
	global img_height
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)
	return zones[y//tmp2,x//tmp1]


def normalize_map():
	global zones
	zones = zones/zones.sum()


def set_random_p():
	global zones
	global image

	for i in range(zones.shape[0]):
		for j in range(zones.shape[1]):
			zones[i,j] = random.random()
	normalize_map()

def generate_uniform_p():
	global zones
	global image
	zones = np.ones((zones.shape[0], zones.shape[1]));
	normalize_map()

if __name__ == '__main__':
	main()


		