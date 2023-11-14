import numpy as np
import pandas as pd
import json

PRS_orig_feature_matrix = np.load('./PRS_feature_matrix_only_ad.npy').astype(np.float32)
usable_samples_ADNI = json.load(open('./usable_samples_ADNI.json'))
final_samples_4yrs=json.load(open('./Final_Samples_4yrs.json'))
final_samples_2yrs=json.load(open('./Final_Samples_2yrs.json'))
covar_df = pd.read_csv('./COVAR_FILE_bigger_dataset.txt', ' ') 
covar_df['AGE'] = covar_df['AGE'] / 100.0

print(usable_samples_ADNI.__len__())
print(final_samples_4yrs.__len__())
print(final_samples_2yrs.__len__())
print(PRS_orig_feature_matrix.shape)

positive_samples = []
negative_samples = []

for i,sample in enumerate(usable_samples_ADNI):
    for s in final_samples_4yrs:
        # print(s)
        if sample == s[0]:
            if s[1] == 0:
                negative_samples.append(PRS_orig_feature_matrix[i])
            else:
                positive_samples.append(PRS_orig_feature_matrix[i])
print(np.min(positive_samples),np.max(positive_samples))
print(np.min(negative_samples),np.max(negative_samples))

import matplotlib.pyplot as plt
plt.hist(positive_samples,10)

# print(PRS_to_AD_mapping.__len__())