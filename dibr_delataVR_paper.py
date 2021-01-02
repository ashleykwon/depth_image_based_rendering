import numpy as np
from PIL import Image
from matplotlib import pyplot as pt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

#read images
file_path = '/Users/ashleykwon/Desktop/depth_map_practice_Nov_2020/'
rgb_original = np.asarray(Image.open(file_path + 'rgb_test.png'))
depth_original = np.asarray(Image.open(file_path + 'depth_test.png'))

#Intrinsic parameters of original camera
K_o = np.array([[1732.87, 0.0, 943.23],[0.0, 1729.90, 548.845040], [0, 0, 1]]); #camera calibrartion matrix
# Extrinsic parameters of original camera
# Rt_o = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]) #rotation and translation matrix

# depth map normalization factors 
# Znear and Zfar are nearest and fartheset points in the scene from the original camera
# Zfar = 2760.510889
# Znear = 34.506386

# Virtual camera parameters
# Intrinsic parameters of virtual camera
#K_v = np.array([[1732.87, 0.0, 943.23], [0.0, 1729.90, 548.845040], [0, 0, 1]])
K_v = np.array([[1732.87, 0.0, 1000.0], [0.0, 1729.90, 548.845040], [0, 0, 1]])
# Extrinsic parameters of virtual camera
#Rt_v = np.array([[1.0, 0.0, 0.0, 1.5924], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]); # rotation and translation matrix
#Rt_v = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1000]]); # rotation and translation matrix
#1.5924

#Converts V_o and D_o to grayscale images
rgb_gray = rgb2gray(rgb_original)
depth_gray = rgb2gray(depth_original)

H, Wid = rgb_gray.shape
x, y = np.meshgrid([x for x in range(Wid)],[y for y in range(H)])

XwYwZw = np.dstack((x, y, depth_gray)).reshape(3, H*Wid)
W = np.linalg.inv(K_o).dot(XwYwZw)
new_proj = K_v.dot(W)


Xc = new_proj[0,:].reshape(Wid, H)
Yc = new_proj[1,:].reshape((Wid, H))
#Zc =  XcYcZc1[2,:].reshape(W, H)

Ix = np.divide(np.multiply(x,Xc.T), depth_gray)
Iy = np.divide(np.multiply(y,Yc.T), depth_gray)


Dx = Ix + rgb_gray
Dy = Iy + rgb_gray

Dx/=max(Dx.flatten())
Dx*=255

Dy/=max(Dy.flatten())
Dy*=255

pt.imsave('Dx3.png', Dx)
pt.imsave('Dy3.png', Dy)

print('done')

