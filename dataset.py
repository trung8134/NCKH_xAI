import os
import cv2 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def array_dataset(img_size):
    def load_images_from_folder(path, dim):
        data = []
        labels = []

        global category 
        category = {'Corn___Common_Rust': 0, 'Corn___Gray_Leaf_Spot': 1, 'Corn___Healthy': 2, 'Corn___Northern_Leaf_Blight': 3, 'Rice___Brown_Spot': 4, 'Rice___Healthy': 5, 'Rice___Leaf_Blast': 6, 'Rice___Neck_Blast': 7, 'Wheat___Brown_Rust': 8, 'Wheat___Healthy': 9, 'Wheat___Yellow_Rust': 10}

        for i in sorted(os.listdir(path)):
            path_class = os.path.join(path, i)
            for j in os.listdir(path_class):
                # Đọc ảnh ở chế độ màu RGB
                img_rgb = cv2.imread(os.path.join(path_class, j), cv2.IMREAD_COLOR)
                img_rgb_resized = cv2.resize(img_rgb, (dim, dim))

                data.append(img_rgb_resized)
                labels.append(category[i]) 

        return data, labels

    dim = img_size
    path = 'Datasets/training'

    x, y = load_images_from_folder(path, dim)
    x = np.array(x)
    y = to_categorical(y, num_classes=len(category)) 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
