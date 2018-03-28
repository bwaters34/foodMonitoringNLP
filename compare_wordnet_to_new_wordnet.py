
from display_html_2 import *
import time
import cPickle as pickle

start_time = time.time()
directory = 'HSLLD/HV1/MT/'
precision_list = []
recall_list = []
settings = (True, False), (False, True), (True, True), (False, False)
for a,b in settings:
    precision, recall, results = evaluate_all_files_in_directory(directory, only_files_with_solutions=True, use_wordnet=a, use_wordnet_food_names=b, )
    true_pos = results.num_true_pos
    false_pos = results.num_false_pos
    false_neg = results.num_false_neg
    false_pos_list = results.false_pos_list
    false_neg_list = results.false_neg_list

    print('precision: {}'.format(precision))
    print('recall {}'.format(recall))
    print('# true pos: {}'.format(true_pos))
    print('# false pos: {}'.format(false_pos))
    print('# false neg: {}'.format(false_neg))
    print('false positives: {}'.format(false_pos_list))
    print('false negatives: {}'.format(false_neg_list))
    print('total runtime: {}'.format(time.time() - start_time))
    precision_list.append(precision)
    recall_list.append(recall)
    # with open('most_recent_results.pickle', 'wb') as f:
    #     pickle.dump(results, f)
for item in list(zip(precision_list, recall_list, settings)):
    print(item)