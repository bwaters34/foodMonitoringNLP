from collections import namedtuple

Accuracy = namedtuple('Accuracy',
						  'num_true_pos num_false_pos num_false_neg false_pos_list false_neg_list')  # makes returning multiple values more clear
