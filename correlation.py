#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd

from collections import defaultdict

from utils import *
from config import X_TRAIN_MP, Y_TRAIN_PATH, UNDER_DIR, MASK_DIR, MODS_DIR


# In[2]:


X_train = read_csv_mp(X_TRAIN_MP, asarray=False)
y_train = pd.read_csv(Y_TRAIN_PATH, header=None)
print('Data loaded.\n')


# # "Vanilla" data (no oversampling, undersampling)

# In[ ]:


##### DELETE #####
#X = X_train.iloc[:,:10000]

print('Calculating X_train corrcoef.')
##### DELETE #####
corr = corrcoef(X_train)
print('Done.\n')

thresholds = [0.7, 0.8, 0.9, 0.95]
for threshold in thresholds:
    print(f"Beginning threshold: {threshold:.2f}")
    mask = corr_filter(corr, threshold=threshold)
    save_mask(mask, 'original', threshold)
    print(f"Threshold complete.\n")


# In[ ]:


for threshold in thresholds:
    mask = np.loadtxt(mask_path('original', threshold), delimiter=',', dtype=int)
    print(f"Threshold: {threshold:.2f}\t\tNumber of features: {np.sum(mask)}")


# # Oversampled data (3 classes)

# In[ ]:


random_state = 0
X_over, _ = oversample(X_train, y=y_train, random_state=random_state)


# In[ ]:


##### DELETE #####
#X_over = X_over.iloc[:,:10000]

print('Calculating X_over corrcoef.')
corr = corrcoef(X_over)
print('Done.\n')

thresholds = [0.7, 0.8, 0.9, 0.95]
for threshold in thresholds:
    print(f"Beginning threshold: {threshold:.2f}")
    mask = corr_filter(corr, threshold=threshold)
    save_mask(mask, 'over', threshold, random_state=random_state)


# In[ ]:


for threshold in thresholds:
    mask = np.loadtxt(mask_path('over', threshold, random_state=random_state), delimiter=',', dtype=int)
    print(f"Threshold: {threshold:.2f}\t\tNumber of features: {np.sum(mask)}")


# # Undersampled data - splits (3 classes)

# In[ ]:


def under_data_path(class_id, split_id, *, random_state):
    return UNDER_DIR / f'random_{random_state}' / f'{class_id}' / f'{split_id}.csv'


# In[ ]:


num_splits = 10
random_state = 0
splits = undersample_split(X_train, y=y_train, num_splits=num_splits, random_state=random_state)


# In[ ]:


thresholds = [0.7, 0.8, 0.9, 0.95]
threshold_to_mask = defaultdict(list)

for i, (X_under, _) in enumerate(splits):
    print(f"Beginning split: {i}")
    
    ##### DELETE #####
    #X_under = X_under.iloc[:,:10000]
    
    print(f'Calculating X_under (split {i}) corrcoef.')
    corr = corrcoef(X_under)
    print('Done.\n')
    
    for threshold in thresholds:
        print(f"Beginning threshold: {threshold}")        
        threshold_to_mask[threshold].append(corr_filter(corr, threshold=threshold))

    print(f"Split complete.\n")

for threshold, mask_list in threshold_to_mask.items():
    save_mask(np.column_stack(mask_list), 'under', threshold, random_state=random_state)


# In[ ]:


for threshold in thresholds:
    mask = np.loadtxt(mask_path('under', threshold, random_state=random_state), delimiter=',', dtype=int)
    mask_all = np.all(mask, axis=1)
    print(f"Threshold: {threshold:.2f}\t\tNumber of features: {np.sum(mask_all)}")


# # Oversampled data (2 classes \[merged smoker + former\])

# In[ ]:


random_state = 0
y_merge = merge_classes(y_train, [1, 2], col=0)
X_over, _ = oversample(X_train, y=y_merge, random_state=random_state)


# In[ ]:


##### DELETE #####
#X_over = X_over.iloc[:,:10000]

print('Calculating X_over corrcoef.')
corr = corrcoef(X_over)
print('Done.\n')

thresholds = [0.7, 0.8, 0.9, 0.95]
for threshold in thresholds:
    print(f"Beginning threshold: {threshold:.2f}")
    mask = corr_filter(corr, threshold=threshold)
    save_mask(mask, 'merged_over', threshold, random_state=random_state)
    print(f"Threshold complete.\n")


# In[ ]:


for threshold in thresholds:
    mask = np.loadtxt(mask_path('merged_over', threshold, random_state=random_state), delimiter=',', dtype=int)
    print(f"Threshold: {threshold:.2f}\t\tNumber of features: {np.sum(mask)}")


# # Undersampled data (2 classes)

# In[ ]:


random_state = 0
y_merge = merge_classes(y_train, [1, 2], col=0)
X_under, _ = undersample(X_train, y=y_merge, random_state=random_state)


# In[ ]:


##### DELETE #####
#X_under = X_under.iloc[:,:10000]

print('Calculating X_under corrcoef.')
corr = corrcoef(X_under)
print('Done.\n')

thresholds = [0.7, 0.8, 0.9, 0.95]
for threshold in thresholds:
    print(f"Beginning threshold: {threshold:.2f}")
    mask = corr_filter(corr, threshold=threshold)
    save_mask(mask, 'merged_under', threshold, random_state=random_state)
    print(f"Threshold complete.\n")


# In[ ]:


for threshold in thresholds:
    mask = np.loadtxt(mask_path('merged_under', threshold, random_state=random_state), delimiter=',', dtype=int)
    print(f"Threshold: {threshold:.2f}\t\tNumber of features: {np.sum(mask)}")


# In[ ]:




