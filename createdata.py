import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import fftpack
quantize50 = [[16,11,10,16],
              [12,12,14,19],
              [14,13,16,24],
              [14,17,22,29]]
path  = f'/mnt/net/filer-ai/shares/proj/pam/data/image/silo/exam/exam.quiz/LCDF000.bmp'

im = Image.open(path)
imarray = np.asarray(im)
#Split image to r,g,b arrays
im_r = imarray[:,:,0]


im_g = imarray[:,:,1]

im_b = imarray[:,:,2]



#reshape to 4x4
def to4x4(arr):
    arr = arr.flatten()
    arr = arr.reshape((-1,4,4))
    return arr

im_r = to4x4(im_r)
im_g = to4x4(im_g)
im_b = to4x4(im_b)



#dct on 4x4 blocks for 3 channels
dct_r = np.empty((129600,4,4))
for i in range(len(im_r)):
    dct_r[i] = fftpack.dct(im_r[i], norm='ortho')
    #quantize here
    dct_r[i] = dct_r[i]/quantize50
    #round
    dct_r[i] = np.round(dct_r[i])

dct_g = np.empty((129600,4,4))
for i in range(len(im_g)):
    dct_g[i]=fftpack.dct(im_g[i], norm='ortho')
    #quantize here
    dct_g[i] = dct_g[i]/quantize50
    #round
    dct_g[i] = np.round(dct_g[i])

dct_b = np.empty((129600,4,4))
for i in range(len(im_b)):
    dct_b[i]=fftpack.dct(im_b[i], norm='ortho')
    #quantize here
    dct_b[i] = dct_b[i]/quantize50
    #round
    dct_b[i] = np.round(dct_b[i])



#reshape to 1x16
dct_r = dct_r.reshape((-1,16))
print(dct_r.shape)
dct_g = dct_g.reshape((-1,16))
dct_b = dct_b.reshape((-1,16))

im_r = im_r.reshape((-1,16))
im_g = im_g.reshape((-1,16))
im_b = im_b.reshape((-1,16))

dct_train = np.vstack((im_r,im_g,im_b))
dct_labels = np.vstack((dct_r,dct_g,dct_b))
print(dct_train.shape)
print(dct_labels.shape)
#np.savetxt('frame39_data.gz', dct_train)
np.savetxt('frame0_quantized_labels.gz', dct_labels)

#dct_r=fftpack.dct(im_r)
#dct_g=fftpack.dct(im_g)
#dct_b=fftpack.dct(im_b)



#inverse dct
# idct_r = fftpack.idct(dct_r)/4096
# idct_g = fftpack.idct(dct_g)/4096
# idct_b = fftpack.idct(dct_b)/4096



#recon = np.dstack((idct_r,idct_g,idct_b))

#Image.fromarray(np.uint8(recon)).show()
