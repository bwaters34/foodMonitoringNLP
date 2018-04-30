
import time
import cPickle as pickle
import matplotlib.pyplot as plt 


thresholds = [x / 10.0 for x in range(0, 10)]
print(thresholds)

from display_html_2 import *
start_time = time.time()
directory = 'HSLLD/HV1/MT/'

precisions = []
recalls = []
for threshold in thresholds:
	precision, recall, results = evaluate_all_files_in_directory(directory, only_files_with_solutions=True, use_wordnet=False, use_wordnet_food_names=True, use_pattern_matching=True, use_word2vec_model = True, use_pretrained_Google_embeddings = True, log_reg_threshold=threshold)

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
	print('recalls:')
	with open('most_recent_results.pickle', 'wb') as f:
	    pickle.dump(results, f)
	
# precisions = [0.759493670886076, 0.7649572649572649, 0.7606837606837606, 0.7672413793103449, 0.7662337662337663, 0.7662337662337663, 0.7662337662337663, 0.7709251101321586, 0.7767857142857143, 0.7844036697247706, 0.8823529411764706]
# recalls = [0.7531380753138075, 0.7489539748953975, 0.7447698744769874, 0.7447698744769874, 0.7405857740585774, 0.7405857740585774, 0.7405857740585774, 0.7322175732217573, 0.7280334728033473, 0.7154811715481172, 0.18828451882845187]


print(precisions)
print(recalls)

plt.plot(recalls, precisions)
# plt.axis([0.0, 1.0, 0.0, 1.0])
for label, x, y in zip(thresholds, recalls, precisions):
	print(x,y)
	plt.annotate(
        s = "{}".format(str(label)),
        xy=(x, y),)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve Over Thresholds of Word2Vec Classifier')
plt.savefig('NEW_VERSION_word2vec_precision_recall.pdf')
