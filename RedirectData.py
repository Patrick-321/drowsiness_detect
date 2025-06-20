#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import shutil
import os
import random

# source
source_folder = './database/yawn_not'
train_folder = './database/yawn_not/train'
val_folder = './database/yawn_not/val'

for folder in [train_folder, val_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

files = [f for f in os.listdir(source_folder) if f.endswith(".jpg") or f.endswith(".png")] 
random.shuffle(files)

split_ratio = 0.8
split_index = int(len(files) * split_ratio)

train_files = files[:split_index]
val_files = files[split_index:]

for filename in train_files:
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(train_folder, filename)
    shutil.move(source_path, destination_path)
    print(f'Moved {filename} to {train_folder}')

for filename in val_files:
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(val_folder, filename)
    shutil.move(source_path, destination_path)
    print(f'Moved {filename} to {val_folder}')

