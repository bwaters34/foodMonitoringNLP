
import time
import cPickle as pickle
import matplotlib.pyplot as plt 
from display_html_2 import *

def plot_precision_recall_curve(precision_1, recall_1,precision_2, recall_2, precision_3, recall_3):
	plt.plot(recall_1, precision_1, label = "System 1")
	plt.legend()
	plt.plot(recall_2, precision_2, label = "System 2")
	plt.legend()
	plt.plot(recall_3, precision_3, label = "System 3")
	plt.legend()
	plt.ylabel("Precision")
	plt.xlabel("Recall")
	plt.show()
	plt.title('Precision Recall Curve Over Different Levenshtein Settings')
	plt.savefig('edit_distance_precision_recall.pdf')


thresholds = [x / 10.0 for x in range(0, 11)]
print(thresholds)

settings = ['system1', 'system2', 'system3']
print(settings)

start_time = time.time()
directory = 'HSLLD/HV1/MT/'
combined_results = []

for setting in settings:
	precisions = []
	recalls = []
	for threshold in thresholds:
		precision, recall, results = evaluate_all_files_in_directory(directory, only_files_with_solutions=True, use_wordnet=False, use_wordnet_food_names=True, use_pattern_matching=True, use_word2vec_model = False, use_edit_distance_matching = True, levenshtein_setting = setting, levenshtein_threshold =threshold )

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
		precisions.append(precision)
		recalls.append(recall)
   	print('precisions:')
   	print(precisions)
	print('recalls:')
	print(recalls)
	combined_results.append((precisions, recalls))
# precisions = [0.759493670886076, 0.7649572649572649, 0.7606837606837606, 0.7672413793103449, 0.7662337662337663, 0.7662337662337663, 0.7662337662337663, 0.7709251101321586, 0.7767857142857143, 0.7844036697247706, 0.8823529411764706]
# recalls = [0.7531380753138075, 0.7489539748953975, 0.7447698744769874, 0.7447698744769874, 0.7405857740585774, 0.7405857740585774, 0.7405857740585774, 0.7322175732217573, 0.7280334728033473, 0.7154811715481172, 0.18828451882845187]


# print(precisions)
# print(recalls)

# plt.plot(recalls, precisions)
# # plt.axis([0.0, 1.0, 0.0, 1.0])
# for label, x, y in zip(thresholds, recalls, precisions):
# 	print(x,y)
# 	plt.annotate(
#         s = "{}".format(str(label)),
#         xy=(x, y),)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision Recall Curve Over Thresholds of Edit ')
# plt.savefig('NEW_VERSION_word2vec_precision_recall.pdf')
print(combined_results)
plot_precision_recall_curve(combined_results[0][0], combined_results[0][1], combined_results[1][0], combined_results[1][1], combined_results[2][0], combined_results[2][1])