"""
Color Visualization
"""
import pdb
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
from scipy import ndimage, misc
import scipy as sp
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import glob
import colorcorrect.algorithm as cca
from skimage import color
from sklearn import svm
from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, neighbors
from sklearn.externals.six import StringIO 
import pydot
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

filesToRead = glob.glob("images/*.data")
print "analyzing files : "+ str(filesToRead) 

contents = [] # list of rgb values
for fn in filesToRead:
	data = open(fn, 'r')
	lines = data.readlines()
	for line in lines:
		[R,G,B] = line.split(',')
		[R,G,B] = [int(R),int(G),int(B)]
		contents.append(np.array([R,G,B]))

contents = np.array(contents)
mean_sample = np.mean(contents,axis=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d',title="RGB")
for sample in contents:
	red = sample[0]
	green = sample[1]
	blue = sample[2]
	ax.scatter(xs=red,ys=green,zs=blue,c='r',marker='o')
plt.title('RGB values')
ax.set_xlabel('RED')
ax.set_ylabel('GREEN')
ax.set_zlabel('BLUE')
ax.scatter(xs=mean_sample[0],ys=mean_sample[1],zs=mean_sample[2],c='b',marker='o')
plt.show()