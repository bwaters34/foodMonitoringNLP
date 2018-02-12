from display_html_old import *

directory = 'HSLLD/HV1/MT/'
precision, recall, true_pos, false_pos, false_neg = evaluate_all_files_in_directory(directory, only_files_with_solutions=True)
print('precision: {}'.format(precision))
print('recall {}'.format(recall))
print('# true pos: {}'.format(true_pos))
print('# false pos: {}'.format(false_pos))
print('# false neg: {}'.format(false_neg))