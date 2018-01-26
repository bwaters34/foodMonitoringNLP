from display_html_2 import *

directory = 'HSLLD/HV1/MT/'
precision, recall, true_pos, false_pos, false_neg = evaluate_all_files_in_directory(directory, only_files_with_solutions=True)
print('precision: {}'.format(precision))
print('recall{}'.format(recall))