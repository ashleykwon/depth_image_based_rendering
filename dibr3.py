import numpy as np
from PIL import Image
from matplotlib import pyplot as pt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

#read images
file_path = 'path'
V_o = np.asarray(Image.open(file_path + 'rgb_test.png'))
D_o = np.asarray(Image.open(file_path + 'depth_test.png'))

#Intrinsic parameters of original camera
#K_o = np.array([[1732.87, 0.0, 943.23],[0.0, 1729.90, 548.845040], [0, 0, 1]]); #camera calibrartion matrix
# Extrinsic parameters of original camera
Rt_o = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]) #rotation and translation matrix

# depth map normalization factors 
# Znear and Zfar are nearest and fartheset points in the scene from the original camera
# Zfar = 2760.510889
# Znear = 34.506386

# Virtual camera parameters
# Intrinsic parameters of virtual camera
#K_v = np.array([[1732.87, 0.0, 943.23], [0.0, 1729.90, 548.845040], [0, 0, 1]])
# Extrinsic parameters of virtual camera
#Rt_v = np.array([[1.0, 0.0, 0.0, 1.5924], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]); # rotation and translation matrix
Rt_v = np.array([[1.0, 0.0, 0.0, 100], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]); # rotation and translation matrix
#1.5924
im1 = rgb2gray(V_o)
#im1 = V_o
im2 = rgb2gray(D_o)

H1, W1 = im1.shape
x1, y1 = np.meshgrid([x for x in range(H1)],[y for y in range(W1)])

H, W = im2.shape
x, y = np.meshgrid([x for x in range(H)],[y for y in range(W)])

fx_d = 1
fy_d = 1

#x1 = x+10
#y1 = y+10

x_k = np.multiply((- im2/fx_d).T, (x1-x))
y_k = np.multiply((- im2/fy_d).T, (y1-y))


#XwYwZw = np.dstack((x.T,y.T,im2))
XwYwZw = np.dstack((x_k.T,y_k.T,im2))
XwYwZw1 = np.hstack((XwYwZw.reshape(H*W,3), np.ones((H*W,1))))

# rotate = np.zeros((3,5))
# rotate[:3, :4] = Rt_o
# rotate[:,4] = Rt_o[:,3].T
# rotate = np.vstack((rotate, np.array([0, 0, 0, 0, 1])))

# rotate2 = np.zeros((3,5))
# rotate2[:3, :4] = Rt_v
# rotate2[:,4] = Rt_v[:,3].T
# rotate2 = np.vstack((rotate2, np.array([0, 0, 0, 0, 1])))

#np.array([0, 0, 1])
#XcYcZc1 = rotate.T.dot(XwYwZw1.T)
#XcYcZc1 = rotate2.T.dot(XwYwZw1.T)
XcYcZc1 = Rt_o.dot(XwYwZw1.T)

Xc = XcYcZc1[0,:].reshape(W, H)
Yc = XcYcZc1[1,:].reshape(W, H)
Zc =  XcYcZc1[2,:].reshape(W, H)

Ix = np.divide(np.multiply(x1,Xc).T, im2)
Iy = np.divide(np.multiply(y1,Yc).T, im2)

Dx = Ix+im1
Dy = Iy+im1

# Dx/=max(Dx.flatten())
# Dx*=255

# Dy/=max(Dy.flatten())
# Dy*=255

pt.imsave('Dx2.png', Dx)
pt.imsave('Dy2.png', Dy)

print('done')

