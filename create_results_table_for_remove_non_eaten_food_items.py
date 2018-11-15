
from display_html_2 import *
import time
import cPickle as pickle

start_time = time.time()
directory = 'HSLLD/HV1/MT/'

old_file_paths = """HSLLD/HV1/MT/geomt1.cha
HSLLD/HV1/MT/kurmt1.cha
HSLLD/HV1/MT/brnmt1.cha
HSLLD/HV1/MT/seamt1.cha
HSLLD/HV1/MT/maymt1.cha
HSLLD/HV1/MT/zenmt1.cha
HSLLD/HV1/MT/tamtp1.cha
HSLLD/HV1/MT/megmt1.cha
HSLLD/HV1/MT/davmt1.cha
HSLLD/HV1/MT/vicmt1.cha
HSLLD/HV2/MT/stnmt2.cha
HSLLD/HV2/MT/ethmt2.cha
HSLLD/HV2/MT/vicmt2.cha
HSLLD/HV2/MT/megmt2.cha
HSLLD/HV2/MT/seamt2.cha
HSLLD/HV2/MT/petmt2.cha
HSLLD/HV2/MT/kurmt2.cha
HSLLD/HV2/MT/zenmt2.cha
HSLLD/HV2/MT/karmt2.cha
HSLLD/HV2/MT/brnmt2.cha
HSLLD/HV2/MT/rosmt2.cha
HSLLD/HV2/MT/diamt2.cha
HSLLD/HV2/MT/catmt2.cha
HSLLD/HV2/MT/inamt2.cha
HSLLD/HV3/MT/melmt3.cha
HSLLD/HV3/MT/ethmt3.cha
HSLLD/HV3/MT/bobmt3.cha
HSLLD/HV3/MT/sarmt3.cha
HSLLD/HV3/MT/deamt3.cha
HSLLD/HV3/MT/kurmt3.cha
HSLLD/HV3/MT/brnmt3.cha
HSLLD/HV3/MT/stnmt3.cha
HSLLD/HV3/MT/tommt3.cha
HSLLD/HV3/MT/rosmt3.cha
HSLLD/HV3/MT/vicmt3.cha
HSLLD/HV3/MT/devmt3.cha
HSLLD/HV3/MT/geomt3.cha
HSLLD/HV3/MT/catmt3.cha
HSLLD/HV5/MT/jammt5.cha
HSLLD/HV5/MT/rosmt5.cha
HSLLD/HV5/MT/ethmt5.cha
HSLLD/HV5/MT/mrkmt5.cha
HSLLD/HV5/MT/brnmt5.cha
HSLLD/HV5/MT/petmt5.cha
HSLLD/HV5/MT/melmt5.cha
HSLLD/HV5/MT/zenmt5.cha
HSLLD/HV5/MT/kurmt5.cha
HSLLD/HV5/MT/bramt5.cha
HSLLD/HV5/MT/shlmt5.cha
HSLLD/HV5/MT/davmt5.cha
HSLLD/HV7/MT/tommt7.cha
HSLLD/HV7/MT/admmt7.cha
HSLLD/HV7/MT/davmt7.cha
HSLLD/HV7/MT/allmt7.cha
HSLLD/HV7/MT/jebmt7.cha""".splitlines(False)

file_paths = """HSLLD/HV1/MT/kurmt1.cha
HSLLD/HV1/MT/seamt1.cha
HSLLD/HV1/MT/maymt1.cha
HSLLD/HV1/MT/zenmt1.cha
HSLLD/HV1/MT/tamtp1.cha
HSLLD/HV1/MT/megmt1.cha
HSLLD/HV1/MT/vicmt1.cha
HSLLD/HV2/MT/stnmt2.cha
HSLLD/HV2/MT/ethmt2.cha
HSLLD/HV2/MT/vicmt2.cha
HSLLD/HV2/MT/megmt2.cha
HSLLD/HV2/MT/seamt2.cha
HSLLD/HV2/MT/petmt2.cha
HSLLD/HV2/MT/kurmt2.cha
HSLLD/HV2/MT/zenmt2.cha
HSLLD/HV2/MT/karmt2.cha
HSLLD/HV2/MT/brnmt2.cha
HSLLD/HV2/MT/rosmt2.cha
HSLLD/HV2/MT/diamt2.cha
HSLLD/HV2/MT/catmt2.cha
HSLLD/HV2/MT/inamt2.cha
HSLLD/HV3/MT/melmt3.cha
HSLLD/HV3/MT/ethmt3.cha
HSLLD/HV3/MT/bobmt3.cha
HSLLD/HV3/MT/sarmt3.cha
HSLLD/HV3/MT/deamt3.cha
HSLLD/HV3/MT/kurmt3.cha
HSLLD/HV3/MT/brnmt3.cha
HSLLD/HV3/MT/stnmt3.cha
HSLLD/HV3/MT/tommt3.cha
HSLLD/HV3/MT/rosmt3.cha
HSLLD/HV3/MT/vicmt3.cha
HSLLD/HV3/MT/devmt3.cha
HSLLD/HV3/MT/geomt3.cha
HSLLD/HV3/MT/catmt3.cha
HSLLD/HV5/MT/jammt5.cha
HSLLD/HV5/MT/rosmt5.cha
HSLLD/HV5/MT/ethmt5.cha
HSLLD/HV5/MT/mrkmt5.cha
HSLLD/HV5/MT/brnmt5.cha
HSLLD/HV5/MT/petmt5.cha
HSLLD/HV5/MT/melmt5.cha
HSLLD/HV5/MT/zenmt5.cha
HSLLD/HV5/MT/kurmt5.cha
HSLLD/HV5/MT/bramt5.cha
HSLLD/HV5/MT/shlmt5.cha
HSLLD/HV5/MT/davmt5.cha
HSLLD/HV7/MT/tommt7.cha
HSLLD/HV7/MT/admmt7.cha
HSLLD/HV7/MT/davmt7.cha
HSLLD/HV7/MT/allmt7.cha
HSLLD/HV7/MT/jebmt7.cha""".splitlines(False)



precision_list = []
recall_list = []

all_false_positives = []
all_false_negatives = []

files_with_solutions = []

remove_non_eaten_food_list = [False, True]
for remove_non_eaten_food in remove_non_eaten_food_list:
    precision, recall, results = evaluate_all_files_in_directory(directory,
                                                                 only_files_with_solutions=True,
                                                                 use_wordnet=False,
                                                                 use_wordnet_food_names=True,
                                                                 use_pattern_matching=True,
                                                                 use_word2vec_model = False,
                                                                 use_pretrained_Google_embeddings=True,
                                                                 file_paths=file_paths,
                                                                 remove_non_eaten_food=remove_non_eaten_food,
                                                                 use_edit_distance_matching=False,
                                                                 base_accuracy_on_how_many_unique_food_items_detected=True)

    true_pos = results.num_true_pos
    false_pos = results.num_false_pos
    false_neg = results.num_false_neg
    false_pos_list = results.false_pos_list
    false_neg_list = results.false_neg_list

    # print('precision: {}'.format(precision))
    # print('recall {}'.format(recall))
    # print('# true pos: {}'.format(true_pos))
    # print('# false pos: {}'.format(false_pos))
    # print('# false neg: {}'.format(false_neg))
    # print('false positives: {}'.format(false_pos_list))
    # print('false negatives: {}'.format(false_neg_list))
    print('total runtime: {}'.format(time.time() - start_time))
    precision_list.append(precision)
    recall_list.append(recall)
    all_false_negatives.append(false_neg_list)
    all_false_positives.append(false_pos_list)

for item in list(zip(precision_list, recall_list)):
    print(item)

print('false positives:')
for group in all_false_positives:
    print(group)
print('false negatives')
for group in all_false_negatives:
    print(group)


print(len(all_false_positives))
print(len(file_paths))