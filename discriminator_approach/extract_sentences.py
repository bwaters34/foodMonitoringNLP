# Written in Python 3.5
# For fuzzy wuzzy see: https://github.com/seatgeek/fuzzywuzzy
import pickle
import os
from fuzzywuzzy import process
from pprint import pprint
import gensim
import time
import nltk


class extract_sentences:
    def __init__(self,
                 hslld,
                 hslld_all_food=None,
                 hslld_eaten_food=None,
                 window_of_sentences=3):
        self.hslld = self.load_files(folder=hslld)
        self.hslld_all_food = self.load_files(folder=hslld_all_food)
        self.hslld_eaten_food = self.load_files(folder=hslld_eaten_food)
        self.window_of_sentences = window_of_sentences
        # print(len(self.hslld), len(self.hslld_all_food),
        #       len(self.hslld_eaten_food))

    def traverse_sentences(self):
        # basis of traversal is HSLLD all eaten foods
        files_to_traverse = self.hslld_all_food
        all_foods = []
        eaten_foods = []
        for fileCount, fileName in enumerate(files_to_traverse, 1):
            # print(fileName)
            hslld_file_all_food_names = "../" + fileName
            file_contents = open(hslld_file_all_food_names,
                                 'r').read().split('\n')
            if file_contents[0] != 'UNIQUE':
                continue
            hslld_file_name = "../" + '/'.join(fileName.split('/')[1:])

            hslld_file_only_eaten_food_names = "../solutions_only_eaten/" + \
                '/'.join(fileName.split('/')[1:])
            if os.path.exists(hslld_file_name):
                print("file exists", fileCount, hslld_file_name)
            else:
                print("FILE DOESNT Exist!!!!", hslld_file_name)
            all_foods, eaten_foods = self.extract_sentences(hslld_file_name=hslld_file_name,
                                                            hslld_file_all_food_names=hslld_file_all_food_names,
                                                            hslld_file_only_eaten_food_names=hslld_file_only_eaten_food_names)
        print("Task Completed!!", len(all_foods), len(eaten_foods))
        self.store_files(all_foods, "list_of_all_food_sentences.pickle")
        self.store_files(eaten_foods, "list_of_eaten_food_sentences.pickle")

    def extract_sentences(self, hslld_file_name, hslld_file_all_food_names, hslld_file_only_eaten_food_names):
        entire_file = open(hslld_file_name, 'r').read().split('\n')
        entire_file = [x for x in entire_file if len(x) > 0]
        entire_file = [x for x in entire_file if x[0] == '*']
        all_food_names = open(hslld_file_all_food_names).read().split('\n')[1:]
        eaten_food_names = open(hslld_file_only_eaten_food_names,
                                'r').read().split('\n')[1:]

        all_food_names = [x.split(':')
                          for x in all_food_names if x != '' or x != ' ']
        all_food_names = [x[1] for x in all_food_names if len(x) > 1]

        eaten_food_names = [x.split(':') for x in eaten_food_names]
        eaten_food_names = [x[1] for x in eaten_food_names if len(x) > 1]

        return self.match_food_keywords(hslld_read_file=entire_file,
                                        list_of_all_food_names=all_food_names,
                                        list_of_eaten_food_names=eaten_food_names)

    def match_food_keywords(self, hslld_read_file, list_of_all_food_names, list_of_eaten_food_names):
        sentences_with_eaten_food = []
        sentences_with_all_food = []
        for line_no, sentence in enumerate(hslld_read_file):
            min_line = max(0, line_no - self.window_of_sentences)
            max_line = min(len(hslld_read_file) - 1,
                           line_no + self.window_of_sentences)
            if len(sentence) == 0:
                continue
            all_food = process.extract(
                query=sentence, choices=list_of_all_food_names)
            if self.check_for_confidence(all_food):
                sentences_with_all_food.append(
                    hslld_read_file[min_line:max_line + 1])
            eaten_food = process.extract(
                query=sentence, choices=list_of_eaten_food_names)
            if self.check_for_confidence(eaten_food):
                # Save to sentences with eaten food
                # print(eaten_food, sentence)
                sentences_with_eaten_food.append(
                    hslld_read_file[min_line:max_line + 1])
        if len(sentences_with_all_food) == 0:
            sentences_with_all_food = None
        if len(sentences_with_eaten_food) == 0:
            sentences_with_eaten_food = None
        return sentences_with_all_food, sentences_with_eaten_food

    def check_for_confidence(self, matches, confidence=80):
        for match in matches:
            if match[1] > confidence:
                return True
        return False

    def load_Google_Word2Vec(self, loc):
        print("Loading Google Word2Vec.......")
        start_time = time.time()
        model = gensim.models.KeyedVectors.load_word2vec_format(
            loc, binary=True)
        end_time = time.time()
        print("Time taken to load model {:.4f} seconds".format(
            end_time - start_time))
        return model

    def clean_text(self, text):
        for sequence_no, sequence in enumerate(text):
            # print("Length of sequence -> ", len(sequence))
            for sentence_no, sentence in enumerate(sequence):
                sentence = ''.join(
                    [x.lower() for x in sentence[4:] if x.isalpha() or x == ' ']).strip()
                sentence = nltk.word_tokenize(sentence)
                text[sequence_no][sentence_no] = sentence
        print(len(text), len(text[0]), len(text[0][0]))

        return text

    def load_files(self, folder):
        with open(folder, 'rb') as f:
            return pickle.load(f)

    def store_files(self, Variable, folder):
        with open(folder, 'wb') as f:
            pickle.dump(Variable, f)


if __name__ == '__main__':
    extract_sent = extract_sentences(hslld='food_files.pickle',
                                     hslld_all_food='food_files_solution.pickle',
                                     hslld_eaten_food='only_eaten_food_files_solution.pickle')
    # extract_sent.traverse_sentences()
    list_of_all_food_sentences = extract_sent.load_files(
        './list_of_all_food_sentences.pickle')
    list_of_eaten_food_sentences = extract_sent.load_files(
        './list_of_eaten_food_sentences.pickle')

    # loc_of_Google_Word2Vec = '~/CCPP/data/wordEmbeddings/GoogleNews-vectors-negative300.bin.gz'
    # word2vec = extract_sent.load_Google_Word2Vec(loc_of_Google_Word2Vec)
    clean_all_food_sentences = extract_sent.clean_text(
        list_of_all_food_sentences)
    eaten_food_sentences = extract_sent.clean_text(
        list_of_eaten_food_sentences)
