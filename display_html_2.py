#https://seaborn.pydata.org/generated/seaborn.distplot.html
import re

from matplotlib.axes._base import _AxesBase
import numpy as np 
import front_end 
import pickle 
import sys 
from collections import defaultdict 
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import pos_tag, word_tokenize
from os import path
from collections import namedtuple
import solution_parser
import CMUTweetTagger
import os
import cal_calorie_given_food_name 
import parse 
import nltk 
import time 
from pyjarowinkler import distance
import levenshtein_distance_customized
import wordnet_explorer
from gensim.models import Word2Vec 
import gensim 
from namedtuples import Accuracy
import phrasemachine

#Word2Vec
use_Google = 0
if use_Google:
	print "Loading Google Pre-Trained Word Embeddings"
	start = time.time()
	# word2vec_filepath = '/home/pritish/CCPP/wordEmbeddings/GoogleNews-vectors-negative300.bin.gz'
	word2vec_filepath = '/home/bwaters/Documents/word2vec/GoogleNews-vectors-negative300.bin.gz'
	model_google = gensim.models.KeyedVectors.load_word2vec_format(word2vec_filepath, binary=True)
	print "Time taken to load google Embeddings", time.time() - start

def load(fileName):
	with open(fileName, 'r') as f:
		return pickle.load(f) 

def save(variable, fileName):
	with open(fileName, 'w') as f:
		pickle.dump(variable, f)

def read_file(fileName, only_files_with_solutions = False, base_accuracy_on_how_many_unique_food_items_detected = True, use_second_column = False, pos_tags_setting = 'nltk', use_wordnet = False, wordnet_setting = 'most_common', use_word2vec_model = False, use_pretrained_Google_embeddings = True, use_edit_distance_matching = False, use_wordnet_food_names = False, use_pattern_matching = True, ):
	"""
	:param fileName: Name of file to be read
	:param parser_type:
	:param only_files_with_solutions: If True, we only bother parsing files with solutions (for example, if we only care about precision and recall, we don't care about
	:param base_accuracy_on_how_many_unique_food_items_detected:
	:param use_second_column: if True, then a subset of phrases/words from the second column thought to be most-foodlike are added to the set of food names. This usually increases recall slightly and tanks precision. See create_extra_food_names() in create_links.py for more details.
	:param pos_tags_setting: if "nltk", then the default nltk perceptron POS tagger is used. If "ark", then the Ark Tweet NLP tagger is used (http://www.cs.cmu.edu/~ark/TweetNLP/).
	:param use_wordnet: if True,
	:return: write2file, a string that is a valid HTML file of the original transcript with food matches highlighted, and results, a namedtuple with attributes num_true_pos, num_false_pos, and num_false_neg
	"""
	# levenshtein_distance_calculator = levenshtein_distance_customized.levenshtein_distance(a=(3, 3, 1),
 #                            e=(3, 3, 1),
 #                            i=(3, 3, 1),
 #                            o=(3, 3, 1),
 #                            u=(3, 3, 1),
 #                            s=(0, 0, 1))
 	levenshtein_distance_calculator = levenshtein_distance_customized.levenshtein_distance(s=(0, 0, 1))
	write2file = ''
	total_calorie = 0.0
	calorie = cal_calorie_given_food_name.food_to_calorie()
	par = parse.parse(pattern=pos_tags_setting)

	#WSD
	unknown_tag = {}
	unknown_tag['unk'] = np.zeros(300)

	if use_word2vec_model:
		if use_pretrained_Google_embeddings:
			start = time.time()
			Word2Vec_model = model_google

			Word2Vec_words = Word2Vec_model.vocab
			model = load('./wsd/LogisticRegression_double_neg_Google_no_data_label')
		else:
			Word2Vec_model = Word2Vec.load('./wsd/word_embeddings_HSLLD.bin')
			Word2Vec_words = list(Word2Vec_model.wv.vocab)
			model = load('./wsd/LogisticRegressionModel_twice_neg')
	

	#Previous versions
	#foodNames = load(path.join('.', path.join('data','food_pair_dict.pickle')))
	#foodNames = load('.\\data\\nltk_food_dictionary.pickle')
	foodNames = load("./data/food_desc_files/food_names.pickle")
	# print('adding extra names')
	Yelena_Mejova_food_names = load("./data/food_desc_files/all_food_words_by_Yelena_Mejova.pickle")
	# foodNames = Yelena_Mejova_food_names
	# Yelena_Mejova_food_names = load("./data/food_desc_files/for_sure_food_words_by_Yelena_Mejova.pickle")
	# foodNames = Yelena_Mejova_food_names
	# print ("Added names by Yelena Mejova")

	# print(len(foodNames))
	if use_second_column:
		extraFoodNames = load("./data/food_desc_files/extra_food_names.pickle")
		foodNames.update(extraFoodNames)
	foodNames.update(Yelena_Mejova_food_names)
	if use_wordnet_food_names:
		wordnet_food_names = load("./data/food_desc_files/wordnet_food_words.pickle")
		banned_words = ['dinner', 'supper', 'lunch', 'breakfast', 'meal', 'dessert', 'food']
		for word in banned_words:
			wordnet_food_names.pop(word)
		foodNames.update(wordnet_food_names)

	# print(len(foodNames))
	foodGroup = load("./data/food_desc_files/food_group.pickle")
	langua = load("./data/food_desc_files/langua.pickle")

	#ark_parsed_data = ark_parser(fileName)

	unique_food_names = {}
	f = file(fileName, 'r')
	current_line_number = 0
	predicted_food_labels_set = set() # syntax: key = (line_number, (start_index_of_food_string_on_line, end_index_of_food_string_on_line), where ending indices are inclusive.
	solution_set_loaded = False
	solution_file_path = path.join('solutions', fileName)
	pos_tags_filename = "pos_tags/" + fileName
	pos_tags_dict = pickle.load(open(
		pos_tags_filename))  # keys are line numbers, values are lists of tuples of (term, type, confidence) where each tuple is a word on the line
	# TODO: let parser support ark pos tags
	try:
		print('loading solution set')
		solution_set = solution_parser.get_solution_set_from_file(solution_file_path)
		solution_set_loaded = True
	except IOError:
		print('no solution file found for: ' + solution_file_path)

	# if we only want files with solutions, and no solution set is found, break early so we don't need to parse the file for food words.
	if only_files_with_solutions:
		if not solution_set_loaded:
			return "solution set not found", None
	for line_no, i in enumerate(f): # i is the current line (a string)
		wsd_i = i
		calorie_text = ''
		food_id_group_pairs = []
		food_id_langua_pairs = []
		current_line_number += 1
		if i[0] == '*':
			#print "LINE NO -> ", line_no
			word_char_index, word_char_index_string_fromat = provide_words_with_char_nos(i, line_no+1)
			# print "LOOK HERE", word_char_index, word_char_index_string_fromat
			text = ''
			edit_distance_i = i
			i = i.lower()
			#i = i.split()
			#for word in i:
			#	if word not in foodNames:
			#		text += word + ' '
			#	else:
			#		text += '<mark>'+word+'</mark> 's
			#write2file += text + '<br>'
			found_at_least = 0
			index_of_food_names = []
			temp_i = re.sub('[^a-zA-Z0-9 \n]', ' ', i[4:])
			#temp_i = i[4:]
			spans_found_on_line = []

			#FOR EDIT DISTANCE
			pos_tags = pos_tags_dict[current_line_number]
			if pos_tags_setting == 'nltk':
				sentence_pos_tags = par.pattern_matching(edit_distance_i, pos_tag(edit_distance_i.split()))
			elif pos_tags_setting == 'ark':
				sentence_pos_tags = par.pattern_matching(edit_distance_i, pos_tags_dict[current_line_number])
			# elif:
			# 	sentence_pos_tags = par.generate_max_two_words(edit_distance_i, pos_tag)				
			else:
				raise ValueError
			# print sentence_pos_tags
			#print "ATTENTION", sentence_pos_tags
			if use_wordnet:
				if len(sentence_pos_tags) > 0:
					# if word == 'carrot':
					# 	print "CARROT"
					# if word == 'tomatoes':
					# 	print "DIAGONISING", sentence_pos_tags, word
					# print('candidates to check:')
					# print(len(sentence_pos_tags))
					for food_data in sentence_pos_tags:
						if len(food_data[1]) <4: continue
						candidate_word = food_data[1]
						# if candidate_word == word:
						# 	continue  # we already guessed it
						if wordnet_explorer.string_is_descendant_of_food(candidate_word, wordnet_setting):
							# print('descended from food: {}'.format(str(food_data)))
							# it might be food!
							index_of_food_names.append([food_data[2], food_data[3]])
							spans_found_on_line.append([food_data[2], food_data[3]])
							found_at_least = 1

			wsd_i = wsd_i[6:]
			wsd_i = wsd_i.split()
			n = 2
			for to_append in xrange(n):
				wsd_i.append("unk")
				wsd_i.insert(0, "unk")



			if use_pattern_matching:
				pos_tags = pos_tags_dict[current_line_number]
				phrases = phrasemachine.ark_get_phrases_wrapper(pos_tags)
				for word in phrases:
					if word in foodNames:
						# print(tags)
						# print word

						# WSD
						if len(word.split()) == 1:
							# WSD applicable
							if use_word2vec_model:
								try:
									if use_pretrained_Google_embeddings:
										print "Step 0 (Using Google Pre-Trained Word Embeddings) ", wsd_i, word
										# wsd_i_temp = [temp_w_for_emb.lower() for temp_w_for_emb in wsd_i]
										wsd_i_temp = ["".join(re.split("[^a-zA-Z]*", temp_w_for_emb.lower())) for
													  temp_w_for_emb in wsd_i]
										# [" ".join(re.split("['a-zA-Z]*", dummy_word)) dummy_word for wsd_i_temp]
										print "Step 0.1", wsd_i_temp, wsd_i, word
										food_place_index = wsd_i_temp.index(word)
										print "Step 1 ", food_place_index

										sent_format = wsd_i[food_place_index - n:food_place_index + n + 1]
										print "Step 2", sent_format
										# sent_word2vec_format = [Word2Vec_model[wsd_word] if wsd_word in Word2Vec_words else unknown_tag['unk'] for wsd_word in sent_format]
										sent_word2vec_format = [
											Word2Vec_model.word_vec(wsd_word) if wsd_word in Word2Vec_words else
											unknown_tag['unk'] for wsd_word in sent_format]
										testing_array = np.asarray(sent_word2vec_format)
										testing_array = testing_array.reshape(1, 1500)
										print "Intermediate step -> ", testing_array.shape
										prediciton = model.predict(testing_array)
										print "Step 3", testing_array.shape, prediciton
										if prediciton == 0:
											print "Predicted not a food", wsd_i, word
											continue

									else:
										print "Step 0", wsd_i, word
										food_place_index = wsd_i.index(word)
										print "Step 1 ", food_place_index
										sent_format = wsd_i[food_place_index - n:food_place_index + n + 1]
										print "Step 2", sent_format
										sent_word2vec_format = [
											Word2Vec_model[wsd_word] if wsd_word in Word2Vec_words else unknown_tag[
												'unk'] for wsd_word in sent_format]
										testing_array = np.asarray(sent_word2vec_format)
										testing_array = testing_array.reshape(1, 500)
										print "Intermediate step -> ", testing_array.shape
										prediciton = model.predict(testing_array)
										print "Step 3", testing_array.shape, prediciton
										if prediciton == 0:
											print "Predicted not a food", wsd_i, word
											continue
										# continue
								except:
									print "Couldn't run WSD", sys.exc_info()

						unique_food_names[word] = 1
						found_at_least = 1

						# #Previous Setting
						# c =  i.find(word)
						# index_of_food_names.append([c, c + len(word) + 1])

						# #removed the plus one
						# spans_found_on_line.append((c, c + len(word)))
						try:
							temp_calorie = calorie.cal_calorie(word)
							total_calorie += temp_calorie
							calorie_text += '<br><mark>' + word + "</mark>-> " + str(temp_calorie)
						except:
							print sys.exc_info()
							pass

						individual_food_words = word.split()
						# for word, label in tags:
						# 	if word == last_word and check_if_noun(label):
						# 		index_of_food_names.append([c, c + len(word) + 1])
						# 		print('chose word: '+ word)
						# 		pass
						# 	else:
						# 		continue
						# print(tags)
						# print(individual_food_words)


						for match in re.finditer(word, i):
							# print "Sentence -> ", temp_i, "matches -> ", match
							food_match_indexes = match.span()
							index_of_food_names.append([food_match_indexes[0], food_match_indexes[1]])
							spans_found_on_line.append([food_match_indexes[0], food_match_indexes[1]])

						# Adding stuffs after reading documentation from USDA
						# print ("food -> ", foodNames[word], foodGroup[foodNames[word]])
						food_id = foodNames[word]
						if food_id in foodGroup:
							food_group_for_food_id = foodGroup[food_id]
							food_id_group_pairs.append([word, food_group_for_food_id])

						if food_id in langua:
							temp_langua = langua[food_id]
							t = []
							for temp_words in temp_langua:
								t.append(temp_words)
							food_id_langua_pairs.append([word + " " + food_id, t])
						# food_id_langua_pairs =
						# print("food -> ", food_id_group_pairs)
						# Checking for EDIT Distance
						if use_edit_distance_matching:
							for food_data in sentence_pos_tags:  # TODO: renable string matching
								# print "Sentence pos tags", sentence_pos_tags
								k1 = float(len(food_data[1])) / float(len(word))
								if 0.6 < k1 and k1 < 1.4:
									# k1 = float(len(food_data[1]))/float(len(word))
									# if 0.6 < k1 and k1 < 1.4:
									# k1 = jaccard_distance(food_data[1], word)
									# if k1 < 0.3:
									# print "Crossed Jaccard Barrier", k1
									# if 0.6 < k and k < 1.4:
									# k1 = abs(len(food_data[1]) - len(word))
									# if k1 <= 3:s
									# if word == 'tomatoes':
									# 	print word, food_data[1], "Reached first pass",  nltk.edit_distance(word, food_data[1])
									# print "yes", food_data[1], word
									# PERFORM EDIT DISTANCE
									if word == food_data[1]: continue
									distance = nltk.edit_distance(word, food_data[1])
									# temp =  " ".join(re.findall("[a-zA-Z]+", food_data[1]))
									# temp2 = " ".join(re.findall("[a-zA-Z]+", word))

									# temp = re.sub('[^a-zA-Z]+', ' ', food_data[1])
									# temp2 = re.sub('[^a-zA-Z]+', ' ', word)



									# temp = ''.join([x if x.isalpha() else ' ' for x in food_data[1]]).strip()
									# temp2 = ''.join([x if x.isalpha() else ' ' for x in word]).strip()

									# Manual checking
									# k2 = 0
									# if len(temp) > 2 and len(temp2) > 2:
									# 	if temp[-1] == 's' or temp2[-1] == 's':
									# 		if temp[:-1] == temp2:
									# 			print "yes if 1", temp[:-1], temp2
									# 			k2 = 1
									# 		elif temp == temp2[:-1]:
									# 			k2 = 1
									# 		else:
									# 			pass
									# 	elif temp == temp2:
									# 		k2 =1
									# 	else:
									# 		pass

									# if len(temp) > 2 and len(temp2) > 2:
									# 	if temp[-2:] == 'es' or temp2[-2:] == 'es':
									# 		if temp[:-2] == temp2:
									# 			k2 = 1
									# 		elif temp == temp2[:-2]:
									# 			k2 = 1
									# 		else:
									# 			pass
									# 	elif temp == temp2:
									# 		k2 =1
									# 	else:
									# 		pass

									# print "check -> ", word, food_data[1], temp, temp2, k1

									# distance = levenshtein_distance_calculator.calculate_distance(temp2, temp)
									# distance = 0
									k2 = distance / float(max(len(word), len(food_data)))
									# if k2  == 1:
									if k2 < 0.25:
										# k2 = 3
										# if distance <= k2:

										# k2 = 3
										# if distance <= k2:

										# k3 = distance.get_jaro_distance(word, food_data[1], winkler = True, scaling = 0.1)
										# if k3 > 0.90:

										found_at_least = 1
										# if word == 'tomatoes':
										# 	print git word, food_data[1], "Reached SECOND pass",  nltk.edit_distance(word, food_data[1])
										index_of_food_names.append([food_data[2], food_data[3]])
										spans_found_on_line.append([food_data[2], food_data[3]])

										with open("./notes/wordnet_twitter_25_per_normalLevenshtien.txt",
												  "a") as myfile:
											# with open("./notes/edit_distance_30_percen.txt", "a") as myfile:

											# with open("./notes/edit_distance_4.txt", "a") as myfile:
											# with open("./notes/edit_distance_jaro.txt", "a") as myfile:

											myfile.write(
												word + "," + food_data[1] + "," + str(distance) + ", " + str(
													k1) + " , " + str(
													k2) + "\n")
										# print "word found", word, len(word), max_len, max_len_word
										# print ("Temproray -> ", temp_i)
										# print ("Final i -> ", i)
			else:
				for word in foodNames:
					if temp_i.__contains__(' ' + word + ' '):
						# print(tags)
						# print word

						#WSD
						if len(word.split()) == 1:
							#WSD applicable
							if use_word2vec_model:
								try:
									if use_pretrained_Google_embeddings:
										print "Step 0 (Using Google Pre-Trained Word Embeddings) ", wsd_i, word
										# wsd_i_temp = [temp_w_for_emb.lower() for temp_w_for_emb in wsd_i]
										wsd_i_temp = ["".join(re.split("[^a-zA-Z]*", temp_w_for_emb.lower())) for temp_w_for_emb in wsd_i]
										# [" ".join(re.split("['a-zA-Z]*", dummy_word)) dummy_word for wsd_i_temp]
										print "Step 0.1", wsd_i_temp, wsd_i, word
										food_place_index = wsd_i_temp.index(word)
										print "Step 1 ", food_place_index

										sent_format = wsd_i[food_place_index-n:food_place_index+n+1]
										print "Step 2", sent_format
										# sent_word2vec_format = [Word2Vec_model[wsd_word] if wsd_word in Word2Vec_words else unknown_tag['unk'] for wsd_word in sent_format]
										sent_word2vec_format = [Word2Vec_model.word_vec(wsd_word) if wsd_word in Word2Vec_words else unknown_tag['unk'] for wsd_word in sent_format]
										testing_array = np.asarray(sent_word2vec_format)
										testing_array = testing_array.reshape(1, 1500)
										print "Intermediate step -> ", testing_array.shape
										prediciton = model.predict(testing_array)
										print "Step 3", testing_array.shape, prediciton
										if prediciton == 0:
											print "Predicted not a food", wsd_i, word
											continue

									else:
										print "Step 0", wsd_i, word
										food_place_index = wsd_i.index(word)
										print "Step 1 ", food_place_index
										sent_format = wsd_i[food_place_index-n:food_place_index+n+1]
										print "Step 2", sent_format
										sent_word2vec_format = [Word2Vec_model[wsd_word] if wsd_word in Word2Vec_words else unknown_tag['unk'] for wsd_word in sent_format]
										testing_array = np.asarray(sent_word2vec_format)
										testing_array = testing_array.reshape(1, 500)
										print "Intermediate step -> ", testing_array.shape
										prediciton = model.predict(testing_array)
										print "Step 3", testing_array.shape, prediciton
										if prediciton == 0:
											print "Predicted not a food", wsd_i, word
											continue
											# continue
								except:
									print "Couldn't run WSD", sys.exc_info()

						unique_food_names[word] = 1
						found_at_least = 1

						# #Previous Setting
						# c =  i.find(word)
						# index_of_food_names.append([c, c + len(word) + 1])

						# #removed the plus one
						# spans_found_on_line.append((c, c + len(word)))
						try:
							temp_calorie = calorie.cal_calorie(word)
							total_calorie += temp_calorie
							calorie_text += '<br><mark>'+word+"</mark>-> "+str(temp_calorie)
						except:
							print sys.exc_info()
							pass

						individual_food_words = word.split()
						# for word, label in tags:
						# 	if word == last_word and check_if_noun(label):
						# 		index_of_food_names.append([c, c + len(word) + 1])
						# 		print('chose word: '+ word)
						# 		pass
						# 	else:
						# 		continue
						# print(tags)
						# print(individual_food_words)


						for match in re.finditer(word, i):
							# print "Sentence -> ", temp_i, "matches -> ", match
							food_match_indexes = match.span()
							index_of_food_names.append([food_match_indexes[0], food_match_indexes[1]])
							spans_found_on_line.append([food_match_indexes[0], food_match_indexes[1]])

						#Adding stuffs after reading documentation from USDA
						#print ("food -> ", foodNames[word], foodGroup[foodNames[word]])
						food_id = foodNames[word]
						if food_id in foodGroup:
							food_group_for_food_id = foodGroup[food_id]
							food_id_group_pairs.append([word, food_group_for_food_id])

						if food_id in langua:
							temp_langua = langua[food_id]
							t = []
							for temp_words in temp_langua:
								t.append(temp_words)
							food_id_langua_pairs.append([word + " " + food_id, t])
							#food_id_langua_pairs =
						# print("food -> ", food_id_group_pairs)
					#Checking for EDIT Distance
					if use_edit_distance_matching:
						for food_data in sentence_pos_tags:  # TODO: renable string matching
							# print "Sentence pos tags", sentence_pos_tags
							k1 = float(len(food_data[1])) / float(len(word))
							if 0.6 < k1 and k1 < 1.4:
								# k1 = float(len(food_data[1]))/float(len(word))
								# if 0.6 < k1 and k1 < 1.4:
								# k1 = jaccard_distance(food_data[1], word)
								# if k1 < 0.3:
								# print "Crossed Jaccard Barrier", k1
								# if 0.6 < k and k < 1.4:
								# k1 = abs(len(food_data[1]) - len(word))
								# if k1 <= 3:s
								# if word == 'tomatoes':
								# 	print word, food_data[1], "Reached first pass",  nltk.edit_distance(word, food_data[1])
								# print "yes", food_data[1], word
								# PERFORM EDIT DISTANCE
								if word == food_data[1]: continue
								distance = nltk.edit_distance(word, food_data[1])
								# temp =  " ".join(re.findall("[a-zA-Z]+", food_data[1]))
								# temp2 = " ".join(re.findall("[a-zA-Z]+", word))

								# temp = re.sub('[^a-zA-Z]+', ' ', food_data[1])
								# temp2 = re.sub('[^a-zA-Z]+', ' ', word)



								# temp = ''.join([x if x.isalpha() else ' ' for x in food_data[1]]).strip()
								# temp2 = ''.join([x if x.isalpha() else ' ' for x in word]).strip()

								#Manual checking
								# k2 = 0
								# if len(temp) > 2 and len(temp2) > 2:
								# 	if temp[-1] == 's' or temp2[-1] == 's':
								# 		if temp[:-1] == temp2:
								# 			print "yes if 1", temp[:-1], temp2
								# 			k2 = 1
								# 		elif temp == temp2[:-1]:
								# 			k2 = 1
								# 		else:
								# 			pass
								# 	elif temp == temp2:
								# 		k2 =1
								# 	else:
								# 		pass

								# if len(temp) > 2 and len(temp2) > 2:
								# 	if temp[-2:] == 'es' or temp2[-2:] == 'es':
								# 		if temp[:-2] == temp2:
								# 			k2 = 1
								# 		elif temp == temp2[:-2]:
								# 			k2 = 1
								# 		else:
								# 			pass
								# 	elif temp == temp2:
								# 		k2 =1
								# 	else:
								# 		pass

									# print "check -> ", word, food_data[1], temp, temp2, k1

								# distance = levenshtein_distance_calculator.calculate_distance(temp2, temp)
								# distance = 0
								k2 = distance/float(max(len(word), len(food_data)))
								# if k2  == 1:
								if k2 < 0.25:
								#k2 = 3
								#if distance <= k2:

									# k2 = 3
									# if distance <= k2:

									# k3 = distance.get_jaro_distance(word, food_data[1], winkler = True, scaling = 0.1)
									# if k3 > 0.90:

									found_at_least = 1
									# if word == 'tomatoes':
									# 	print git word, food_data[1], "Reached SECOND pass",  nltk.edit_distance(word, food_data[1])
									index_of_food_names.append([food_data[2], food_data[3]])
									spans_found_on_line.append([food_data[2], food_data[3]])

									with open("./notes/wordnet_twitter_25_per_normalLevenshtien.txt", "a") as myfile:
									# with open("./notes/edit_distance_30_percen.txt", "a") as myfile:

										# with open("./notes/edit_distance_4.txt", "a") as myfile:
										# with open("./notes/edit_distance_jaro.txt", "a") as myfile:

										myfile.write(
											word + "," + food_data[1] + "," + str(distance) + ", " + str(k1) + " , " + str(
												k2) + "\n")
									# print "word found", word, len(word), max_len, max_len_word
									# print ("Temproray -> ", temp_i)
									# print ("Final i -> ", i)

			if found_at_least:	
				dic = minimum_no_meeting_rooms(index_of_food_names, len(i))
				#print('dic')
				#print(dic)
				for char_pos in dic:
					if dic[char_pos] == 1:
						text += '<mark>' +  i[char_pos] + '</mark>'
					else:
						text += i[char_pos]
				text += calorie_text

				tuples_list = give_largest_non_overlapping_sequences(spans_found_on_line)  # filters out spans that conflict with other spans. larger spans are given priority
				for tup in tuples_list:
					set_elem = (current_line_number, tup) # add line number so we know where in the document we got it
					predicted_food_labels_set.add(set_elem)

			else:
				pass
				text += i[1:] 

	write2file += "<hr>" + "Total Calories -> " + str(total_calorie) 
	num_true_pos = None # give dummy values in case try fails
	num_false_pos = None
	num_false_neg = None
	if solution_set_loaded:
		print('loading solution set')
		solution_set = solution_parser.get_solution_set_from_file(solution_file_path)
		print('calculating')
		if base_accuracy_on_how_many_unique_food_items_detected:
			food_names_only_solution_set = solution_parser.convert_solution_set_to_set_of_food_names(fileName, solution_set)
			food_names_only_predicted_set = solution_parser.convert_solution_set_to_set_of_food_names(fileName, predicted_food_labels_set)
			precision, recall, false_pos_list, false_neg_list, true_pos_list = solution_parser.calculate_precision_and_recall(
				food_names_only_solution_set, food_names_only_predicted_set)
		else:
			precision, recall, false_pos_list, false_neg_list, true_pos_list = solution_parser.calculate_precision_and_recall(solution_set, predicted_food_labels_set)
		num_true_pos = len(true_pos_list)
		num_false_pos = len(false_pos_list)
		num_false_neg = len(false_neg_list)
		print('file:' + fileName)
		print('precision: ' +  str(precision))
		print('recall: ' +  str(recall))
		print('true positives:') + str(true_pos_list)
		if not base_accuracy_on_how_many_unique_food_items_detected:
			for line in solution_parser.get_corresponding_lines(fileName, true_pos_list):
				print(line)
		print('false positives: ' + str(false_pos_list))
		if not base_accuracy_on_how_many_unique_food_items_detected:
			for line in solution_parser.get_corresponding_lines(fileName, false_pos_list):
				print(line)
		print('false negatives: ' + str(false_neg_list))
		if not base_accuracy_on_how_many_unique_food_items_detected:
			for line in solution_parser.get_corresponding_lines(fileName, false_neg_list):
				print(line)
		print('# true pos: {}'.format(num_true_pos))
		print('# false pos: {}'.format(num_false_pos))
		print('# false neg: {}'.format(num_false_neg))

		if not base_accuracy_on_how_many_unique_food_items_detected:
			write2file += '<br><hr>'+"Precision: "+str(precision)+ \
							"<br>Recall: "+str(recall) + "<br><hr>"
			write2file += 	"False Positives<br>"+ str(false_pos_list)+ \
							"<br>"
			for line in solution_parser.get_corresponding_lines(fileName, false_pos_list):
				write2file += str(line)+ " ---> <mark>" + str(line[1][line[0][1][0]:line[0][1][1]]) +"</mark><br>"
			write2file +=	"<hr>False negatives:<br>"+str(false_neg_list) + "<br>"
			for line in solution_parser.get_corresponding_lines(fileName, false_neg_list):
				write2file += str(line)+ " ---> <mark>" + str(line[1][line[0][1][0]:line[0][1][1]]) +"</mark><br>"

	else:
		print('no solution set found')
	#return write2file, unique_food_names
	#namedtuple()

	results = Accuracy(num_true_pos=num_true_pos, num_false_pos=num_false_pos, num_false_neg=num_false_neg, false_pos_list=false_pos_list, false_neg_list=false_neg_list)


	return write2file, results


def provide_words_with_char_nos(sentence, line_no):
	temp_char = ''
	start_count = 0 
	return_array = []
	for index, char in enumerate(sentence):
		if char != ' ' and char != '\t':
			temp_char += char 
		else:
			return_array.append([temp_char, start_count, index])
			start_count = index + 1 
			temp_char = ' '

	#Converting to displayable format (String format)
	return_string = '<br>(line->' + str(line_no)+ ") "
	for word in return_array:
		return_string += word[0].lower() + " ("+str(word[1])+","+str(word[2])+") "
	return_string += "<br>"
	return return_array, return_string


def jaccard_distance(word1, word2):
	# word1 = list(set(word1))
	# word2 = list(set(word2))
	# print word1, word2
	word1_and_word2 = set(word1).intersection(word2)
	# print word1_and_word2
	word1_or_word2 = set(word1).union(word2)
	# print word1_or_word2
	return float(len(word1_and_word2))/float(len(word1_or_word2))

def join_tags(sentence):
	text = '     '
	for i in sentence:
		text += '(' + i[0] + "->" + i[1] + ") "
	return text

def match_word(food_key_word, sentence, value = 0):
	food_key_word = food_key_word.split()
	sentence = sentence.split()
	for word in food_key_word:
		if word not in sentence:
			return 0
	return 1 

def minimum_no_meeting_rooms(list_of_timings, length_of_sent):
	dic = defaultdict(int)
	for i in xrange(1, length_of_sent):
		dic[i] = 0
	for meeting_schedules in list_of_timings:
		for i in xrange(meeting_schedules[0], meeting_schedules[1]):
			dic[i] = 1 
	return dic

def check_if_noun(tag):
	if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
		return True
	return False

def give_largest_non_overlapping_sequences(list_of_start_end_tuples):
	Sequence = namedtuple('Sequence', ['start', 'end', 'size'])
	list_of_named_sequences = [Sequence(start = x[0], end = x[1], size = x[1] - x[0] - 1) for x in list_of_start_end_tuples] # size is -1 because the end number represents the index of the character AFTER the last character in the sequence.
	sorted_by_size_sequences = sorted(list_of_named_sequences, key=lambda seq: seq.size) # smallest size is first, largest size is last
	non_overlapping_sequences = []
	while len(sorted_by_size_sequences) > 0:
		sequence = sorted_by_size_sequences.pop() # last element in list, therefore sequence with largest size still on the list
		if not conflicts_with_sequences(non_overlapping_sequences, sequence):
			non_overlapping_sequences.append(sequence)
	extracted_tuples = [(seq.start, seq.end) for seq in non_overlapping_sequences]
	return extracted_tuples

def conflicts_with_sequences(list_of_sequences, test_sequence):
	"""Tests if test_sequence conflicts with any sequence in the list_of_sequences"""
	for already_added_sequence in list_of_sequences:
		if sequences_overlap(already_added_sequence, test_sequence):
			return True
	return False

def sequences_overlap(seq1, seq2):
	"""Returns if two sequences overlap"""
	if seq1.end <= seq2.start: # seq1 must end before seq2 begins. they do not overlap
		return False
	elif seq2.end <= seq1.start:
		return False
	else:
		return True

def ark_parser(fileName):
	final_list_of_sentences = []
	list_of_sentences = open(fileName, "r").read()
	for sentence in list_of_sentences.split('\n'):
		if len(sentence) > 1:
			if sentence[0] == '*':
				final_list_of_sentences.append(' '.join(sentence.split()))
	print final_list_of_sentences
	var = CMUTweetTagger.runtagger_parse(final_list_of_sentences)
	return var

def evaluate_all_files_in_directory(directory_path, only_files_with_solutions = False, base_accuracy_on_how_many_unique_food_items_detected = True, use_second_column = False, pos_tags_setting = 'ark', use_wordnet = True, wordnet_setting = 'most_common',  use_word2vec_model = False, use_pretrained_Google_embeddings = True, use_edit_distance_matching = False, use_wordnet_food_names = False, use_pattern_matching=False):
	parameters_used = locals() # locals returns a dictionary of the current variables in memory. If we call it before we do anything, we get a dict of all of the function parameters, and the settings used._
	sum_true_pos = 0
	sum_false_pos = 0
	sum_false_neg = 0
	list_of_false_pos_lists = []
	list_of_false_neg_lists = []
	for filename in os.listdir(directory_path):
		file_path = directory_path + '/' + filename
		print(file_path)
		html_format, results = read_file(file_path, only_files_with_solutions=only_files_with_solutions,  base_accuracy_on_how_many_unique_food_items_detected=base_accuracy_on_how_many_unique_food_items_detected, use_second_column=use_second_column, pos_tags_setting=pos_tags_setting, use_wordnet=use_wordnet, wordnet_setting=wordnet_setting, use_word2vec_model=use_word2vec_model, use_pretrained_Google_embeddings=use_pretrained_Google_embeddings, use_edit_distance_matching=use_edit_distance_matching, use_wordnet_food_names = use_wordnet_food_names, use_pattern_matching=use_pattern_matching)
		if results is not None: # there wasn't a solution set for that file
			if results.num_true_pos is not None:  # if it is none, a solution set was not loaded
				sum_true_pos += results.num_true_pos
			if results.num_false_pos is not None:
				sum_false_pos += results.num_false_pos
			if results.num_false_neg is not None:
				sum_false_neg += results.num_false_neg
			if results.false_pos_list is not None:
				list_of_false_pos_lists.append(results.false_pos_list)
			if results.false_pos_list is not None:
				list_of_false_neg_lists.append(results.false_neg_list)
	combined_results = Accuracy(num_true_pos=sum_true_pos, num_false_pos=sum_false_pos, num_false_neg=sum_false_neg, false_pos_list=list_of_false_pos_lists, false_neg_list=list_of_false_neg_lists)
	precision = sum_true_pos / float(sum_true_pos + sum_false_pos)
	recall = sum_true_pos / float(sum_true_pos + sum_false_neg)
	print(parameters_used)
	return precision, recall, combined_results


if __name__ == '__main__':
	try:
		#fileName = 'HSLLD/HV3/MT/brtmt3.cha' # coffee
		
		start = time.time()
		fileName = 'HSLLD/HV1/MT/conmt1.cha'
		html_format, results = read_file(fileName)
		#print "HTNL Format", html_formatf
		print "Time taken to run the script", time.time() -start
		front_end.wrapStringInHTMLWindows(body = html_format)

	except:
		print "none"
		print sys.exc_info()
	# print jaccard_distance("pritish", "pritish yu")
	# fileCounts = []
	# all_files = load("C:\\Users\\priti\\OneDrive\\Documents\\CCPP\\FoodMonitoring-NLP\\data\\food_files.pickle")
	# c = 0
	# for file_name in all_files:
	# 	print "File ", c 
	# 	c += 1 
	# 	try:
	# 		html_format, count = read_file(file_name)
	# 	except:
	# 		continue
	# 	else:
	# 		fileCounts.append(len(cont))
	# sns.distplot(fileCounts, 
	# 				#hist = False,
	# 				kde = False, 
	# 				#rug=False, 
	# 				norm_hist = False, 
	# 				rug_kws={"color": "g"}, 
	# 				kde_kws={"color": "k", "lw": 3, "label": "KDE"},
	# 				hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"})
	# plt.show()

#786.390255213 secs
