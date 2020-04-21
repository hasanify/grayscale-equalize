from IPython.display import display, Math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('sample.jpg')

plt.figure(num="Original Histogram")
img = np.asarray(img)
flat = img.flatten()
plt.hist(flat, bins=50)
plt.show()

display(Math(r'P_x(j) = \sum_{i=0}^{j} P_x(i)'))

def get_histogram(image, bins):
    histogram = np.zeros(bins)
    
    for pixel in image:
        histogram[pixel] += 1

    return histogram

hist = get_histogram(flat, 256)
# plt.plot(hist)

def sum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

cs = sum(hist)

display(Math(r's_k = \sum_{j=0}^{k} {\frac{n_j}{N}}'))

nj = (cs - cs.min()) * 255
N = cs.max() - cs.min()

cs = nj / N

cs = cs.astype('uint8')

img_new = cs[flat]

plt.figure(num="Equalized Histogram")

plt.hist(img_new, bins=50)
img_new = np.reshape(img_new, img.shape)
img_new
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

fig.add_subplot(1,2,1)
plt.imshow(img, cmap='gray')

fig.add_subplot(1,2,2)
plt.imshow(img_new, cmap='gray')

plt.show(block=True)