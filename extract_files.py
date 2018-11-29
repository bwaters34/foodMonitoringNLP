from collections import defaultdict

import os
import sys
import nltk
import pickle
import matplotlib.pyplot as plt
import numpy as np

# start_file = '.\HSLLD'
start_file = 'solutions_only_eaten'
# print os.walk(start_file)
directory = [x[0] for x in os.walk(start_file) if x[0][-2:] == 'MT']
print(directory)
overall_food_presence = []
file_directory = []
# print directory


def open_file(fileName):
    with open(fileName, 'r') as f:
        return pickle.load(f)


def save_file(fileName, Variable):
    with open(fileName, 'wb') as f:
        pickle.dump(Variable, f)


# dictionary = open_file("foodNames.pickle")
#dictionary = open_file("food_Name_1.pickle")
# print('test')
for files in directory:
    # print('adsad')
    for f in os.listdir(files):
        # print('test')
        with open(files + '/' + f) as file:
            print files + '/' + f
            file_directory.append(files + '/' + f)
print(file_directory)
save_file('only_eaten_food_files_solution.pickle', file_directory)
# '''
# food_presence = 0
# unique = defaultdict(float)
# for i in file:
# 	if i[0] == '*':
# 		i = nltk.word_tokenize(i)
# 		for word in i:
# 			if word in dictionary and word not in unique:
# 				#print word, dictionary[word], file
# 				unique[word] = 1
# 				food_presence += 1
# overall_food_presence.append(food_presence)
# #'''
# print overall_food_presence, len(overall_food_presence)
# #overall_food_presence = [10, 10, 2, 2, 3, 3, 3, 3]
# overall_food_presence.sort()
# plt.hist(overall_food_presence, 10, rwidth = 0.25)
# plt.ylabel("Frequency of such docs")
# plt.xlabel("Time food keyword appears in doc")
# plt.show()
#save_file("food_files.pickle", file_directory)
#food = open_file("food_files.pickle")
# rint food == file_directory
