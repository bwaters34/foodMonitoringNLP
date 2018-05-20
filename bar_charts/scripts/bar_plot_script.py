import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 9
precision = (75.27, 75.00, 58.38, 58.25, 62.5, 74.78, 76.37, 77.15, 77.05)
recall = (58.33, 63.75, 72.50, 75, 73.75, 72.91, 75.41, 74.54, 74.49)
f1_score = (65.73, 68.92, 64.68, 65.57, 67.68, 73.83, 75.89, 75.84, 75.75)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups) * 2.0
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, precision, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Precision')
 
rects2 = plt.bar(index + bar_width, recall, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Recall')

rects3 = plt.bar(index + bar_width + bar_width, recall, bar_width ,
                 alpha=opacity,
                 color='r',
                 label='F1_score')
 
plt.xlabel('System')
plt.ylabel('Percentage')
plt.title('Precision Vs Recall')
plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
plt.xticks(index + bar_width, ('USDA', 'Twitter', 'WordNet',
								'Pluralization', 'Phrase Machine', 'Banned Words',
								'Span Merging', 'WSD', 'Levenshtein'))
plt.legend()
 
# plt.tight_layout()
plt.show()