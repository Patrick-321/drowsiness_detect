#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset

dataset = load_dataset('MichalMlodawski/closed-open-eyes')
# dataset = load_dataset('dataset_name', split='train')


# In[ ]:




# In[3]:


from datasets import DatasetDict, Dataset
import numpy as np

full_train_dataset = dataset['train']

split_dataset = full_train_dataset.train_test_split(test_size=0.2, seed=42) 

train_dataset = split_dataset['train']
validation_dataset = split_dataset['test']

validation_test_split = validation_dataset.train_test_split(test_size=0.5, seed=42)  

validation_dataset = validation_test_split['train']
test_dataset = validation_test_split['test']

dataset_split = DatasetDict({
    'train': train_dataset,
    'val': validation_dataset,
    'test': test_dataset
})

print("Training size:", len(dataset_split['train']))
print("Validation size:", len(dataset_split['val']))
print("Test size:", len(dataset_split['test']))


# In[6]:


dataset_split['val'].select(range(500))


# In[7]:


dataset_split['train']


# In[5]:


import os

import cv2

def coco_to_yolo(x,y,w,h,width,height):
    return [((2*x + w)/(2*width)) , ((2*y + h)/(2*height)), w/width, h/height]

def preprocessing(parititon: str, data: object):
    os.makedirs(f"datasets/images/{parititon}", exist_ok=True)
    os.makedirs(f"datasets/labels/{parititon}", exist_ok=True)

    data = data[parititon].select(range(5000))

    for i, sample in enumerate(data):
        # if(sample['Label'] == 'open_eyes'):
        #     labels = [0,1]
        #     text_image = "open"
        if(sample['Label'] != 'closed_eyes'):
            continue
        labels = [0,0]
        text_image = "close"
        img = sample['Image_data']['file']
        bboxes = [sample['Left_eye_react'], sample['Right_eye_react']]
        width = int(sample['Image_data']['file'].size[0])
        height = int(sample['Image_data']['file'].size[1])

        items = []
        for label, box in zip(labels,bboxes):
            xc,yc,w,h = coco_to_yolo(box[0],box[1],box[2],box[3],width,height)
            items.append(f"{label} {xc} {yc} {w} {h}")

        with open(f"datasets/labels/{parititon}/{i}_{text_image}.txt", "w") as f:
            for item in items:
                f.write(f"{item}\n")

        img.save(f"datasets/images/{parititon}/{i}_{text_image}.png")


# In[6]:


preprocessing("train", dataset_split)
preprocessing("val", dataset_split)
preprocessing("test", dataset_split)

