import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2

images = os.path.join("dataset/images")
annotations = os.path.join("dataset/annotations")
train = pd.read_csv(os.path.join("dataset/train.csv"))
submission = pd.read_csv(os.path.join("dataset/submission.csv"))

print(len(train))
print(train.head())

print(len(submission))
print(submission.head())

print(len(os.listdir(images)))

a = os.listdir(images)
a.sort()
b = os.listdir(annotations)
b.sort()
print(len(b))

train_images = a[1698:]
test_images = a[:1698]
print(len(train_images), len(test_images))

options = train['classname'].unique()
train.sort_values('name', axis=0, inplace=True)

bbox = []
for i in range(len(train)):
    arr = []
    for j in train.iloc[i][['x1', 'x2', 'y1', 'y2']]:
        arr.append(j)
    bbox.append(arr)

train["bbox"] = bbox

def get_boxes(id):
    boxes = []
    for i in train[train['name'] == str(id)]['bbox']:
        boxes.append(i)
    return boxes

# print(get_boxes(train_images[3]))
# image=train_images[3]

# img = cv2.imread(os.path.join(images,image))

# boxes = get_boxes(image)

# for box in boxes:
#     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
# cv2.imshow("img", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

img_size = 224


def create_data():
    data=[]
    for i in range(len(train)):
            arr=[]
            for j in train.iloc[i]:
                arr.append(j)
            
            print(arr)
            img_array=cv2.imread(os.path.join(images,arr[0]))
            
            crop_image = img_array[arr[2]:arr[4],arr[1]:arr[3]]
            try:
                new_img_array=cv2.resize(crop_image,(img_size,img_size))
            except:
                continue
            data.append([new_img_array,arr[5]])
    return data