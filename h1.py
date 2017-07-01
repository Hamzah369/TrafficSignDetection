import cv2
import numpy as np

SZ=20
bin_n = 16 
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)    
    return hist

svm=cv2.ml.SVM_load('svm_data.dat')
img=cv2.imread('01.png')
test=hog(img)
test1 = np.float32(test).reshape(-1,64)
y=svm.predict(test1)

print(y)

