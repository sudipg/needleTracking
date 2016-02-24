"""
Color Classificaton
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


def is_metallic(RGB):
	"""
	takes in [R,G,B] sample, returns True or False if metal
	"""
	mean_sample = np.array([ 228.48076923 , 231.34615385 , 229.34615385])
	tolerance = 35;
	return np.linalg.norm(np.array(RGB) - mean_sample) <= tolerance

def is_metallic_fast(img_RGB):
	mean_sample = np.array([ 228.48076923 , 231.34615385 , 229.34615385])
	tolerance = 40;
	return np.linalg.norm(img_RGB - mean_sample, axis=2)<=tolerance