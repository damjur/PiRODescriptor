
# coding: utf-8


from bs4 import BeautifulSoup
import urllib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

import os
import shutil

from descriptor import extract, distance, D

# In[42]:


HOWMANY = 2
MAX_PATCH_SIZE = 64
N_POINTS = 2
DEBUG = False
URL = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html"


# In[29]:


def generatePointsForOne(image):
	height,width,_ = image.shape
	height -= MAX_PATCH_SIZE/2
	width -= MAX_PATCH_SIZE/2
	ys = np.random.randint(MAX_PATCH_SIZE/2,height,size=(N_POINTS))
	xs = np.random.randint(MAX_PATCH_SIZE/2,width,size=(N_POINTS))
	return [np.dstack((ys,xs))[0]]

def generatePoints(images):
	points = []
	for image in images:
		points += generatePointsForOne(image)
	return np.array(points)

def getRotats(images,points):
	r = []
	ones = np.ones(shape=(N_POINTS, 1))
	
	print("Generating rotated images")
	for a in tqdm([0,1,2,3,5,10,15,30,60,90,180]):
		for img,pp in zip(images,points):
			COLS,ROWS,_ =  img.shape
			M = cv2.getRotationMatrix2D((COLS//2,ROWS//2),a,1)
			cos = np.abs(M[0, 0])
			sin = np.abs(M[0, 1])
 
			# compute the new bounding dimensions of the image
			nW = int((COLS * sin) + (ROWS * cos))
			nH = int((COLS * cos) + (ROWS * sin))
 
			# adjust the rotation matrix to take into account translation
			M[0, 2] += (nW / 2) - ROWS//2
			M[1, 2] += (nH / 2) - COLS//2
			
			rimg = cv2.warpAffine(img,M,(nW,nH))
			rpp = M.dot(np.hstack([pp, ones]).T).T
			r   += [{"img":rimg,"points":rpp}]
	return r
			

def getScales(images,points):
	r = []
	
	print("Generating scaled images")
	for img,pp in zip(images,points):
		r   += [{"img":img,"points":pp}]
		
	for s in tqdm(np.arange(1.25,3.75,0.25)):
		for img,pp in zip(images,points):
			rimg = cv2.resize(img,None,fx=s,fy=s)
			rpp = s*pp
			r   += [{"img":rimg,"points":rpp}]
	return r

def getBlures(images,points):
	r = []
	
	print("Generating blured images")
	for img,pp in zip(images,points):
		r   += [{"img":img,"points":pp}]
	
	for b in tqdm(np.arange(1,10)):
		for img,pp in zip(images,points):
			rimg = cv2.GaussianBlur(img,(5,5),b)
			r   += [{"img":rimg,"points":pp}]
	return r

def getJpgeds(images,points):
	r = []
	dirname = "tmp"
	label = "file{}_{}.jpeg"
	if os.path.isdir(dirname) and DEBUG:
		shutil.rmtree(dirname) 
	os.mkdir(dirname)
	
	print("Generating images compressed using JPEG algorithm")
	for q in tqdm(np.arange(0,101,10)):
		for i,(img,pp) in enumerate(zip(images,points)):
			filename = os.path.join(dirname, label.format(i,q))
			cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, i])
			rimg = cv2.imread(filename)
			r   += [{"img":rimg,"points":pp}]
	shutil.rmtree(dirname) 
	return r   

def getTransformations(images,points):
	x = {}
	x["rotat"] = getRotats(images,points)
	x["scale"] = getScales(images,points)
	x["blure"] = getBlures(images,points)
	x["jpged"] = getJpgeds(images,points)
	return x


# In[30]:


def getData(t):
	epsilon = 0.05
	X = {}
	Y = {}
	
	if "blure" in t:
		rds = []
		print("Extracting descriptors (blured)")
		for i in tqdm(t["blure"]):
			rds += [extract(i["img"],i["points"])]
		r0 = 0
		rx  = np.arange(0,10)
		
		rds = np.array(rds).reshape((HOWMANY,len(rx),N_POINTS,D))
		rd0 = rds[r0,:]
		ry  = np.zeros(len(rx))
		print("Extracting distances (blured)")
		for i,img in tqdm(enumerate(rds)):
			cimg = rd0[i]
			for k,angle in enumerate(img):
				for j,d1 in enumerate(cimg):
					row = []
					for d2 in angle:
						row += [distance(d1,d2)]
					ry[k] += (np.array(row) <= row[j]*(1+epsilon)).sum() - 1
		X["blure"] = rx
		Y["blure"] = ry
	
	if "jpged" in t:
		rds = []
		print("Extracting descriptors (JPEG compression)")
		for i in tqdm(t["jpged"]):
			rds += [extract(i["img"],i["points"])]
		r0 = 0
		rx  = np.arange(0,101,10)
		rds = np.array(rds).reshape((HOWMANY,len(rx),N_POINTS,D))
		rd0 = rds[r0,:]
		ry  = np.zeros(len(rx))
		print("Extracting distances (JPEG compression)")
		for i,img in tqdm(enumerate(rds)):
			cimg = rd0[i]
			for k,angle in enumerate(img):
				for j,d1 in enumerate(cimg):
					row = []
					for d2 in angle:
						row += [distance(d1,d2)]
					ry[k] += (np.array(row) <= row[j]*(1+epsilon)).sum() - 1
		X["jpged"] = rx
		Y["jpged"] = ry
		
	if "rotat" in t:
		r0 = 0
		rds = []
		print("Extracting descriptors (rotation)")
		for i in tqdm(t["rotat"]):
			rds += [extract(i["img"],i["points"])]
		rx  = [0,1,2,3,5,10,15,30,60,90,180]
		rds = np.array(rds).reshape((HOWMANY,len(rx),N_POINTS,D))
		rd0 = rds[r0,:]
		ry  = np.zeros(len(rx))
		print("Extracting distances (rotation)")
		for i,img in tqdm(enumerate(rds)):
			cimg = rd0[i]
			for k,angle in enumerate(img):
				for j,d1 in enumerate(cimg):
					row = []
					for d2 in angle:
						row += [distance(d1,d2)]
					ry[k] += (np.array(row) <= row[j]*(1+epsilon)).sum() - 1
		X["rotat"] = rx
		Y["rotat"] = ry
	
	if "scale" in t:
		rds = []
		print("Extracting descriptors (scale)")
		for i in tqdm(t["scale"]):
			rds += [extract(i["img"],i["points"])]
		r0 = 0
		rx  = np.arange(1.25,3.75,0.25)
		rx = np.append([1],rx)
		rds = np.array(rds).reshape((HOWMANY,len(rx),N_POINTS,D))
		rd0 = rds[r0,:]
		ry  = np.zeros(len(rx))
		print("Extracting distances (scale)")
		for i,img in tqdm(enumerate(rds)):
			cimg = rd0[i]
			for k,angle in enumerate(img):
				for j,d1 in enumerate(cimg):
					row = []
					for d2 in angle:
						row += [distance(d1,d2)]
					ry[k] += (np.array(row) <= row[j]*(1+epsilon)).sum() - 1
		X["scale"] = rx
		Y["scale"] = ry
	
	
	return X,Y

def drawPlots(X,Y,L):
	filename = "plot_{}.png"
	for key in ["rotat","scale","blure","jpged"]:
		if key in X and key in Y and key in L:
			plt.figure()
			plt.xlabel(L[key])
			plt.ylabel("Liczba niepoprawnie określonych odległości w optymistycznem przypadku")
			plt.plot(X[key],Y[key],'go--')
			plt.savefig(filename.format(key))


# In[31]:


def loadImage(url):
	raw = urllib.request.urlopen(url).read()
	npraw= np.array(bytearray(raw),dtype=np.uint8)
	return cv2.imdecode(npraw,-1)#-1 -> as is (with the alpha channel)

def getImageName(url):
	return url.split('/').pop().split('.').pop(0)

def loadImagesFromSite(url):
	imgs = []
	print("Loading {} image(s) from {}".format(HOWMANY,url))
	with urllib.request.urlopen(url) as response:
		html = BeautifulSoup(response.read(),"lxml")
		for link in tqdm(html.find_all('a')[:HOWMANY]):
			img = loadImage(link.get('href'))
			imgs += [img]
	return np.array(imgs)  

def saveDataset(t):
	print("Saving dataset")
	for key in tqdm(["rotat","scale","blure","jpged"]):
		with open('pickledDataset_{}'.format(key),'wb') as f:
			pickle.dump(t[key],f)
		
def loadDataset():
	try:
		t = {}
		print("Loading dataset")
		for key in tqdm(["rotat","scale","blure","jpged"]):
			with open('pickledDataset_{}'.format(key),'rb') as f:
				t[key] = pickle.load(f)
		return t
	except:
		print("Load failed")
		return None
	
def getDataset():
	t = loadDataset()
	if t is None or DEBUG:
		images = loadImagesFromSite(URL)
		points = generatePoints(images)
		t = getTransformations(images,points)
		
		saveDataset(t)
	return t
	
def display(X,lset,howmany=None):
	if howmany is None:
		howmany = len(X[lset])
	
	print("Preparing images to display")
	for i in tqdm(range(howmany)):
		img = X[lset][i]["img"]
		pp  = X[lset][i]["points"]
		c = (img.max(),img.max(),img.max())
		plt.figure()
		plt.imshow(img)
	plt.show()	
		


# In[32]:


if __name__ =="__main__":
	t = getDataset()
	
	# display(t,"rotat")
	
	X,Y = getData(t)
	L = {}
	L["rotat"] = "Kąt rotacji (stopnie)"
	L["scale"] = "Przeskalowanie"
	L["blure"] = "Odchylenie standardowe"
	L["jpged"] = "Jakość JPEG"
	drawPlots(X,Y,L)
	
	
