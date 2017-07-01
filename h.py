import cv2
import numpy as np

SZ=20
bin_n = 16 
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
train_list = []
response_list = []
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

for i in range (0,2):
        for j in range (0,3):    
            img=cv2.imread(str(i)+str(j)+'.png')
            data=hog(img)
            np.float32(data).reshape(-1,64)
            train_list.append(data)
            response_list.append(i)

samples = np.float32(train_list)
labels = np.array(response_list)
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(samples, cv2.ml.ROW_SAMPLE,labels)
svm.save('svm_data.dat')


