import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 8
precision = (75.27, 75.00, 58.38, 58.25, 62.5, 74.78, 76.37, 77.15)
recall = (58.33, 63.75, 72.50, 75, 73.75, 72.91, 75.41, 74.54)
f1_score = (65.73, 68.92, 64.68, 65.57, 67.68, 73.83, 75.89, 75.84)
# f1_score = ('USDA', 'Twitter', 'WordNet',
#             'Pluralization', 'Phrase Machine', 'Banned Words',
#             'Span Merging', 'WSD')
fig, ax = plt.subplots()
# ax.scatter(precision, recall)

# evenly sampled time at 200ms intervals
t = np.arange(n_groups)

# red dashes, blue squares and green triangles
ax.plot(t, precision, 'r--', t, recall, 'bs--', t, f1_score, 'g^--')


# for index, text in enumerate(f1_score):
#         ax.annotate(text, (precision[index], recall[index]))
plt.title('Precision Vs Recall')
plt.xlabel('Precision')
plt.ylabel('Recall')
# plt.legend()
# plt.tight_layout()
plt.show() 

# # create plot
# fig, ax = plt.subplots()
# index = np.arange(n_groups) * 2.0
# bar_width = 0.35
# opacity = 0.8
 
# rects1 = plt.bar(index, precision, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  label='Precision')
 
# rects2 = plt.bar(index + bar_width, recall, bar_width,
#                  alpha=opacity,
#                  color='g',
#                  label='Recall')

# rects3 = plt.bar(index + bar_width + bar_width, recall, bar_width ,
#                  alpha=opacity,
#                  color='r',
#                  label='F1_score')
 

# plt.title('Precision Vs Recall')
# plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
# plt.xticks(index + bar_width, ('USDA', 'Twitter', 'WordNet',
# 								'Pluralization', 'Phrase Machine', 'Banned Words',
# 								'Span Merging', 'WSD', 'Levenshtein'))
# plt.legend()
 
# # plt.tight_layout()
# plt.show()