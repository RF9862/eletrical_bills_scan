import cv2
import os 
path = "Reliance Electricity"
lists = os.listdir(path)

for i, im in enumerate(lists):
    img = cv2.imread(os.path.join(path, im))
    h, w = img.shape[0:2]
    cv2.imwrite(os.path.join(path, f"Reliance_{i}.jpg"), img[0:int(h/2)])