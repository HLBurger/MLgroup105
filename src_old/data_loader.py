import kagglehub
import pandas as pd
import os
from preprocessing import *
import matplotlib.pyplot as plt
import cv2

class data():

    def __init__(self):
        self.data = ""

    def data_loader(self, dataset = "UTK", preprocess = True):
        if dataset.upper() == "UTK":
            path = kagglehub.dataset_download("roshan81/ageutk")
            path = path + "\\" + os.listdir(path)[0]
            data = pd.read_csv(path)

        else:
            ValueError("Invalid dataset.")

        if preprocess:
            data = add_path(data, path)
            data = add_labels(data)
        
        self.data = data

    def preview_image(self):
        num = input("Select the image you want to show.")
        image = cv2.imread(data.path[num])
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
test = data()
test.data_loader()
test.preview_image()

