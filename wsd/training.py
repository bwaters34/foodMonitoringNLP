from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
import numpy as np
from pprint import pprint
import time
from random import shuffle
from create_word_embeddings_for_hslld import wordEmbeddings
import re
import os
from display_html_2 import read_file
import solution_parser

class training_classifier:
    def __init__(self, Embeddings, use_Google_pre_Trained_Embeddings=0):
        self.use_Google_Embeddings = use_Google_pre_Trained_Embeddings
        self.unknown_tag = {}
        self.unknown_tag['unk'] = np.zeros(Embeddings.dimension_size)
        # print self.unknown_tag.shape
        self.Embeddings = Embeddings
        self.raw_sentences = self.Embeddings.all_sentences()
        self.food_words = self.Embeddings.food_words_database()

        # Word Embeddings
        start = time.time()
        self.Word2Vec_model = self.Embeddings.sentences_to_wordEmbeddings(
            google_word_embeddings=self.use_Google_Embeddings)
        print "time to upload Embeddings into RAM", time.time() - start

        # print type(self.Word2Vec_model['apple'])
        self.generate_training_data(self.raw_sentences)
        start = time.time()
        self.pos_data, self.neg_data, self.data = self.load_training_data()
        new_data = self.pos_data + self.neg_data[0:2 * len(self.pos_data)]
        print "Time taken to load", time.time() - start
        self.check(new_data)

    # self.check(self.data)

    def generate_training_data(self, sentences, n=2):
        self.dataset_pos = []
        self.dataset_neg = []
        self.dataset = []
        self.data_X = []
        self.data_Y = []

        directory_path = '../HSLLD/HV1/MT/'
        for filename in os.listdir(directory_path):
            file_path = directory_path + '/' + filename
            print(file_path)
            html_format, results, predicted_spans, found_solution = read_file(file_path, use_word2vec_model=False, use_wordnet=False, use_wordnet_food_names=True, use_pattern_matching=True)
            if not found_solution:  # don't use ones we are using as a solution set
                predicted_spans = list(predicted_spans) # previously was a set
                spans_and_lines = solution_parser.get_corresponding_lines(file_path, predicted_spans)
                spans_dict = {} # keys are line numbers, values are lists of spans on the line
                for line, span in spans_and_lines:
                    line_number = span[0]
                    solution_span = span[1]
                    if line_number in spans_dict:
                        spans_dict[line_number].append(solution_span)
                    else:
                        spans_dict[line_number] = [solution_span]
            with open(file_path) as f:
                lines = f.readlines()
            for line_number, line in enumerate(lines, start=1): # as lines start on line 1
                matches = [(m.group(0), (m.start(), m.end())) for m in re.finditer(r'\S+', line)] # example: on input "This is a sentence", returns [('This', (0, 4)), ('is', (5, 7)), ('a', (8, 9)), ('sentence', (10, 18)).
                word_is_predicted = [0 * len(matches)] # list of zeroes
                if line_number in spans_dict:
                    spans_on_current_line = matches[1]
                    list_of_predicted_spans = spans_dict[line_number]
                    for i, curr_span in enumerate(spans_on_current_line):
                        for pred_span in list_of_predicted_spans:
                            curr_start, curr_end = curr_span
                            pred_start, pred_end = pred_span
                            if not (curr_end <= pred_start) and not (pred_end <=curr_start): # must be overlapping
                                word_is_predicted[i] = 1
                                break # skip to next span in spans_on_current_line

            # TOD0: idea: if span is [6:15], do [0:6].split(), [6:15].split(), [15:20].split() where only the middle one is labeled 1?
                sent = matches[1]
                for to_append in xrange(n):
                    sent.append("unk")
                    sent.insert(0, "unk")
                # print sent
                for index_j in xrange(n, len(sent) - n):
                    y_val = 0
                    original_index = index_j - 2
                    sent_format = sent[index_j - n:index_j + n + 1]

                    if word_is_predicted[original_index] == 1:
                        y_val = 1
                        sent[index_j] = 'EmptyWordHereZeroEmbedding'
                        self.dataset_pos.append([sent_format, y_val])
                    else:
                        y_val = 0
                        self.dataset_neg.append([sent_format, y_val])

                    self.dataset.append([sent_format, y_val])
                # 	print sent[index_j]," ",
                # print ""
        self.Embeddings.save("Dataset_pos_without_20_labels_no_context", self.dataset_pos)
        self.Embeddings.save("Dataset_neg_without_20_labels_no_context", self.dataset_neg)
        self.Embeddings.save("Dataset_without_20_labels_no_context", self.dataset)

    # pprint(self.dataset)
    def load_training_data(self):
        pos = self.Embeddings.from_this_folder_load("Dataset_pos_without_20_labels")
        neg = self.Embeddings.from_this_folder_load("Dataset_neg_without_20_labels")
        data = self.Embeddings.from_this_folder_load("Dataset_without_20_labels")
        return pos, neg, data

    def check(self, training_data, split=0.7):
        train_x, train_y = [], []
        full_dataset = []
        for data, y in training_data:
            temp = []
            if self.use_Google_Embeddings:
                # model.Word2Vec_model['rice']
                data = [" ".join(re.split("[^a-zA-Z]*", temp_w_for_emb.lower())) for temp_w_for_emb in data]
                temp = [
                    self.Word2Vec_model.word_vec(word) if word in self.Word2Vec_model.vocab else self.unknown_tag['unk']
                    for word in data]
            else:
                temp = [self.Word2Vec_model[word] if word != 'unk' else self.unknown_tag['unk'] for word in data]
            train_x.append(temp)
            train_y.append(y)
        # full_dataset.append([temp, y])
        # shuffle(full_dataset)
        # training_data = np.asarray(full_dataset[0:int(len(full_dataset)*split)])
        # testing_data = np.asarray(full_dataset[int(len(full_dataset)*split):])
        # # print "ful dataset", full_dataset[0]
        # train_x = training_data[:, 0]
        # train_y = training_data[:, 1]
        # # print "train x", train_x[0], "y", train_y[0]
        # test_x = testing_data[:, 0]
        # test_y = testing_data[:, 1]
        # print training_data.shape, train_x.shape, train_y.shape
        # print testing_data.shape, test_x.shape, test_y.shape
        # logistic = LogisticRegression()
        # logistic.fit(train_x, train_y)
        # predicited = logistic.predict(test_x)
        # pritn metrics.classification_report(test_y, predicited)

        # print y
        # if y == 1:
        # 	print y


        # # Working code for logistic LogisticRegression
        train_x = np.asarray(train_x)
        print train_x.shape
        train_x = train_x.reshape(train_x.shape[0], -1)
        train_y = np.asarray(train_y)
        X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=42)
        logistic = LogisticRegression()

        start = time.time()
        print X_train.shape, Y_train.shape
        logistic.fit(X_train, Y_train)
        predicited = logistic.predict(X_test)
        print metrics.accuracy_score(Y_test, predicited)
        print metrics.classification_report(Y_test, predicited)
        print logistic.score(X_test, Y_test)
        self.Embeddings.save("LogisticRegression_double_neg_Google_no_data_label_aggressive", logistic)
        print "Saved Google Embeddings on double negative data without 20 label dataset and also comparing cases like <apple"

        # print train_x[0]
        print train_x.shape, train_y.shape
        print "Time taken for the prediction ", time.time() - start
        start = time.time()
        predicited = cross_validation.cross_val_predict(LogisticRegression(), train_x, train_y)
        print metrics.accuracy_score(train_y, predicited)
        print metrics.classification_report(train_y, predicited)
        print "Time taken for the prediction ", time.time() - start


if __name__ == '__main__':
    Embeddings = wordEmbeddings()
    classifier = training_classifier(Embeddings, 1)