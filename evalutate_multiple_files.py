
from display_html_2 import *

import time

start_time = time.time()
directory = 'HSLLD/HV1/MT/'
precision, recall, results = evaluate_all_files_in_directory(directory, only_files_with_solutions=True)

true_pos = results.num_true_pos
false_pos = results.num_false_pos
false_neg = results.num_false_neg


print('precision: {}'.format(precision))
print('recall {}'.format(recall))
print('# true pos: {}'.format(true_pos))
print('# false pos: {}'.format(false_pos))
print('# false neg: {}'.format(false_neg))
print('false positives: '.format(results.false_pos_list))
print('false negatives: '.format(results.false_neg_list))
print('total runtime: {}'.format(time.time() - start_time))