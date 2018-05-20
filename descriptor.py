
# coding: utf-8


# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.color as color
from skimage.filters import gaussian, scharr
from skimage.draw import circle,line,circle_perimeter
from skimage.transform import resize
import skimage
import cv2
import os
import pickle

D = 5


# In[32]:

def getCircle(image,radius):
	mask = np.zeros(image.shape)
	mask[circle(image.shape[0]/2,image.shape[1]/2,radius)] = [1,1,1]
	circled = image*mask
	#czy to po prostu nie zeruje zer?
	#DJ: nie, to usuwa puste wiersze/kolumny
	circled = circled[:,~np.all(np.all(circled==0,axis=2),axis=0)]
	circled = circled[~np.all(np.all(circled==0,axis=2),axis=1),:]
	# radius = np.ceil(radius)
	radius = 32
	return resize(circled,(2*radius,2*radius,3)),radius

def getPatch(image,center):
	half_side = 32
	return image[int(center[0])-half_side:int(center[0])+half_side,int(center[1])-half_side:int(center[1])+half_side]

def getRadius(patch):
	#return 32 
	#naprawiłem, jakieś kombinowanie tutaj może raczej tylko zaszkodzić, skoro mamy gwarancję, że tle się zmieści zawsze
	#DJ: usunąłem rzeczoną poprawkę - tu chodzi o skalę 
	epsilon = 0.0005
	side = np.min([patch.shape[0],patch.shape[1]])/2
	edges = scharr(patch[:,:,1])
	p = np.where(edges<=epsilon)
	Ys = p[0] - side
	Xs = p[1] - side
	Rs = Ys**2 + Xs**2
	Rs = Rs[Rs>16]
	if len(Rs)>0:
		r = np.floor(np.sqrt(Rs.min()) * 3)
		return np.min([side,r])
	else:
		return side

def getRadiallySymetricalPoints(k1,k2,radius):
	filename = "points{}_{}".format(k1,k2)
	if os.path.isfile(filename):
		with open(filename,'rb') as f:
			s = pickle.load(f)
	else:
		radii = 0.8*np.random.random(k1) + 0.1
		radii = np.array([np.full((k2),i) for i in radii]).reshape((k2*k1))
		tmp = np.arange(k2)/k2
		thetas = np.array([(np.random.random() + tmp) for i in range(k1)]).reshape((k2*k1))
		thetas *= 2*np.pi
		ys1 = radii * np.sin(thetas)
		xs1 = radii * np.cos(thetas)
		ys2 = radii * np.sin(np.pi + thetas)
		xs2 = radii * np.cos(np.pi + thetas)
		s   = np.dstack((ys1,xs1,ys2,xs2))[0]+1
		
		with open(filename,'wb') as f:
			pickle.dump(s,f)
	return (radius*s).astype(np.int8)

def binarizeImage(image,points):
	return np.array([0 if image[y1,x1]<image[y2,x2] else 1 for y1,x1,y2,x2 in points])

def gradientDes(img):
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
#	 plt.figure()
#	 plt.imshow(laplacian)
	
def rectangleDes(img):
	r=len(img)
	im=np.zeros((r//2,len(circle_perimeter(r//2,r//2,r//2)[0]),3))
	for i in range(r//2):
		im[i] = resize(img[circle_perimeter(r//2,r//2,i)], im.shape[1:])
#	 plt.figure()
#	 plt.imshow(im)
	dMax = np.zeros((r//2,3),dtype=int)
	dMin = np.zeros_like(dMax)
	eMax = np.zeros((len(im[0]),3),dtype=int)
	eMin = np.zeros_like(eMax)
	for i in range(r//2):
		dMax[i] = [np.argmax(im[i,:,x]) for x in range(3)]
		dMin[i] = [np.argmin(im[i,:,x]) for x in range(3)]
	for i in range(len(im[0])):
		eMax[i] = [np.argmax(im[:,i,x]) for x in range(3)]
		eMin[i] = [np.argmin(im[:,i,x]) for x in range(3)]
	return dMax, dMin, eMax, eMin

def findDiff(d1,d2,s):
	nd1 = np.zeros_like(d1)
	nd2 = np.zeros_like(d2)
	x = 0
	for i in range(len(nd1)):
		nd1[i] = d1[int(x)]
		nd2[i] = d2[int(x)]
		x+=s
	s = np.sum(np.absolute(d1-nd2))
	return min(s, np.sum(np.absolute(nd1-d2)))
	
def compareRectangleDes(d1, d2):
	print(d1)
	print()
	print(d2)

	print(d1[5])
	m=min(d1.min(), d2.min())
	d1 -= m
	d2 -= m
	m=max(d1.max(), d2.max())	
	r=np.sum(np.absolute(d1-d2))
	d=m/3*len(d1)*len(d1[0])
	for z in np.linspace(0.65,1,7):
		for i in range(m):
			d1 = (d1+1)%m
			s = findDiff(d1,d2,z)
			if s < r:
				r = s
	return min(r/d,1)

def distanceHLPR(r1,r2):
	return np.abs(r1-r2).sum()

def compareTwoBinaryRings(r1,r2):
	h1,w1 = r1.shape[:2]
	h2,w2 = r2.shape[:2]
	
	if w1!=w2 and h1!=h2:
		return np.inf
	r_max = 0
	d_max = 0
	for i in range(h1):
		d_max += distanceHLPR(r1[i],r2[i])
	
	for r in range(1,len(r1) + 1):
		d = 0
		for i in range(h1):
			d += distanceHLPR(r1[i],np.roll(r2[i],r))
		if d<d_max:
			d_max = d
			r_max = r
	return d_max/w1/h1


# In[33]:


def extractForKeypoint(image,keypoint):
	k = 15
	patch = getPatch(image,keypoint)
	radius = getRadius(patch)
	circled,radius = getCircle(patch,radius)
	points = getRadiallySymetricalPoints(3,12,radius)
	binary0 = binarizeImage(circled[...,0],points)
	binary1 = binarizeImage(circled[...,1],points)
	binary2 = binarizeImage(circled[...,2],points)
	rMax,rMin,eMax,eMin = rectangleDes(circled)
	gradientDes(circled)
	return [np.array([binary0, binary1, binary2]), rMax, rMin, eMax, eMin]
	# return {0:np.array([binary0, binary1, binary2]), 1:rMax, 2:rMin, 3:eMax, 4:eMin}

def extract(image, keypoints):
	blur_sigma = 3
	image = color.rgb2hsv(image)
	#rozmywać coraz mocniej od środka?
	blurred = gaussian(image,sigma=blur_sigma)
	blurred = (blurred - blurred.mean())/blurred.std()#normalizacja
	blurred[blurred<=-3] = -3#usuwanie outlierów
	blurred[blurred>=3] = 3
	blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min())#normalizacja taka, by można to było wyświetlić
	return [extractForKeypoint(blurred,keypoint) for keypoint in keypoints]

def distance(descriptor1, descriptor2):
	dist = 0
	dist+=compareTwoBinaryRings(descriptor1[0], descriptor2[0])
	for i in range(1,2):
		print(i)
		dist+=compareRectangleDes(descriptor1[i], descriptor2[i])
	return dist/5
	
if __name__=="__main__":
	image = data.astronaut()[:-1,:-1]
	plt.imshow(image)
	w=extract(image,[[100,100],[121,101],[100,99]])
	# print("binary",	compareTwoBinaryRings(w[0][0],w[1][0]), compareTwoBinaryRings(w[0][0],w[2][0]))
	# print("rectangle", compareRectangleDes(w[0][1], w[1][1]), compareRectangleDes(w[0][1], w[2][1]))
	# print("rectangle", compareRectangleDes(w[0][2], w[1][2]), compareRectangleDes(w[0][2], w[2][2]))
	# print("rectangle", compareRectangleDes(w[0][3], w[1][3]), compareRectangleDes(w[0][3], w[2][3]))
	
	

