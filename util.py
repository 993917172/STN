import cv2
import numpy as np

DIMS = (400, 400)
CAT1 = 'cat1.jpg'
CAT2 = 'cat2.jpg'

img1 = cv2.imread(CAT1)
img2 = cv2.imread(CAT2)

img1 = np.array(cv2.resize(img1, DIMS)) / 255.0
img2 = np.array(cv2.resize(img2, DIMS)) / 255.0

t_img1 = np.array(img1 * 255.0, dtype=np.uint8)
t_img2 = np.array(img2 * 255.0, dtype=np.uint8)
cv2.imshow('img1', t_img1)
cv2.waitKey()
cv2.imshow('img2', t_img2)
cv2.waitKey()

img1 = np.reshape(img1, (1, 400, 400, 3))
img2 = np.reshape(img2, (1, 400, 400, 3))
img = np.concatenate((img1, img2), axis = 0)

num_batch, H, W, C = np.shape(img)
M = np.array([[1, 0, 0.5], [0, 1, 0.5]])
#M = np.array([[0.707, -0.707, 0.], [0.707, 0.707, 0.]])
M = np.resize(M, (num_batch, 2, 3))

#generate samplegrid
x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
x_s, y_s = np.meshgrid(x, y)
z_s = np.ones(np.prod(x_s.shape))
sample_grid = np.vstack((x_s.flatten(), y_s.flatten(), z_s))
sample_grid = np.resize(sample_grid, (num_batch, 3, H*W))

#apply the transformation matrix to the sample grid
batch_grid = np.matmul(M, sample_grid)
batch_grid = np.reshape(batch_grid, (num_batch, 2, H, W))
batch_grid = np.transpose(batch_grid, (0, 2, 3, 1))

#using interpolation sample the resulting grid from the original image
x = batch_grid[..., 0]
y = batch_grid[..., 1]
x = (x + 1) * W * 0.5
y = (y + 1) * H * 0.5
x0 = np.floor(x).astype(np.int64)
x1 = x0 + 1
y0 = np.floor(y).astype(np.int64)
y1 = y0 + 1

x0 = np.clip(x0, 0, W-1) # [b, h, w]
x1 = np.clip(x1, 0, W-1)
y0 = np.clip(y0, 0, H-1)
y1 = np.clip(y1, 0, H-1)

Ia = img[np.arange(num_batch)[:,None,None], y0, x0] # [b, h, w, c]
Ib = img[np.arange(num_batch)[:,None,None], y0, x1]
Ic = img[np.arange(num_batch)[:,None,None], y1, x0]
Id = img[np.arange(num_batch)[:,None,None], y1, x1]

wa = (x1 - x) * (y1 - y)
wb = (x - x0) * (y1 - y)
wc = (x1 - x) * (y - y0)
wd = (x - x0) * (y - y0)

wa = np.expand_dims(wa, axis = 3)
wb = np.expand_dims(wb, axis = 3)
wc = np.expand_dims(wc, axis = 3)
wd = np.expand_dims(wd, axis = 3)

out = wa * Ia + wb * Ib + wc * Ic + wd * Id

out = np.array(out * 255, dtype=np.uint8)
cv2.imshow('img3', out[0])
cv2.waitKey()
cv2.imshow('img4', out[1])
cv2.waitKey()




