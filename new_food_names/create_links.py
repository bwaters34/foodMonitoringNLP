import os 
import pickle 
import re 
from pprint import pprint 
from sklearn.feature_extraction.text import TfidfVectorizer


class create_links:
	def __init__(self, path):
		self.folder = path
		print (os.listdir(self.folder))
		self.food_group = "FD_GROUP.txt"
		self.language_desc = "LANGDESC.txt"
		self.lanuage_of_food = "LANGUAL.txt"
		self.food_desc = "FOOD_DES.txt"

	def create_food_desc(self):
		self.dic_food_group_final = {}
		self.dic_food_name_to_id = {}
		f = open(self.folder + "/" + self.food_desc)
		for lines in f:
			lines = lines.strip().split('^')
			lines = [x[1:-1] for x in lines]
			food_id = lines[0]
			food_group = lines[1]
			food_name = lines[2]
			self.dic_food_group_final[food_id] = self.dic_food_group[food_group]
			#print(food_id, food_group, self.dic_food_group[food_group])
			#print (food_id, self.create_lists(food_name.lower()))
			food_name = self.create_lists(food_name.lower())
			for name in food_name:
				if name not in self.dic_food_name_to_id:
					self.dic_food_name_to_id[name] = food_id
				else:
					self.dic_food_name_to_id[name] = food_id

		self.save(self.dic_food_group_final, "food_group.pickle")
		self.save(self.dic_food_name_to_id, "food_names.pickle")
		temp_re = self.retrieve("food_names.pickle")
		#print temp_re
		#temp_re = self.retrieve("food_group.pickle")
		#sprint temp_re
		#print self.dic_food_group_final

	def create_extra_food_names(self, column = 2, threshold = 0.6):
		dict = self.add_some_names_from_column(column, threshold)
		self.save(dict, "extra_food_names.pickle")

	def add_some_names_from_column(self, column = 2, threshold = 0.6):
		"""Saves a dictionary of food names from a column (default: 2nd column)"""
		f = open(self.folder + "/" + self.food_desc)
		food_names = []
		food_names_as_lists = []
		line_index = 0
		for lines in f:
			lines = lines.strip().split('^')
			lines = [x[1:-1] for x in lines]
			food_as_list = lines[2].split(',')
			food_names_as_lists.append((food_as_list, line_index))
			line_index += 1
			food_name = ' '.join(food_as_list) # split on comma, join on space to create one cohesive string
			food_names.append(food_name)
			print(food_name)
		tfidf = TfidfVectorizer()
		matrix = tfidf.fit_transform(food_names) # each column corresponds with a word in the vocab, each row is a document
		vocab_dict = tfidf.vocabulary_

		print(matrix._shape)
		index = column -1
		print(food_names_as_lists)
		# if our food name list has enough elements (i.e. if we want the third element in the description, this food name
		# has length >= 3), add it with its line number to the list.
		food_names_from_specified_column = [(x[0][index], x[1]) for x in food_names_as_lists if len(x[0]) >= column]
		whitespace_removed = [(x[0].strip(), x[1]) for x in food_names_from_specified_column]
		single_words_only = [x for x in whitespace_removed if len(x[0].split()) <= 1]
		food_names_dict = {}
		print(len(single_words_only))
		for word, index in single_words_only:

			try: # TODO: This is a hack! Problem: TF-IDF vectorizer tokenizes words using whitespace but also hyphens and other things, while we only use whitespace. This means we get keyerrors on certain words
				col = vocab_dict[word]
				value = matrix[index,col]
			except KeyError:
				continue
			if value > threshold:
				food_names_dict[word] = None # dummy value
		return food_names_dict

	def create_lists(self, sentence):
		temp = []
		sentence = sentence.strip().split(',')
		print sentence
		temp.append(sentence[0])
		if len(sentence) > 1:
			#temp.append(sentence[1].strip())
			pass
			
		if len(sentence) > 1:
			for words in sentence[1:]:
				temp.append(words.strip() + " " + sentence[0].strip())
		return temp 

	def create_language_desc(self):
		self.dic_language_desc = {}
		f = open(self.folder + "/" + self.language_desc, "r")
		for lines in f:
			lines = lines.strip().split('^')
			lines[0] = lines[0][1:-1]
			lines[1] = lines[1][1:-1]
			self.dic_language_desc[lines[0]] = lines[1]
		#print self.dic_language_desc

	def create_lanuage_of_food(self):
		self.dic_lanuage_of_food = {}
		self.temp_dic_lanuage_of_food = {}
		f = open(self.folder + "/" + self.lanuage_of_food, "r")
		for lines in f:
			lines = lines.strip().split('^')
			lines[0] = lines[0][1:-1]
			lines[1] = lines[1][1:-1]
			if lines[0] not in self.dic_lanuage_of_food:
				self.dic_lanuage_of_food[lines[0]] = []

				self.dic_lanuage_of_food[lines[0]].append(lines[1])
			else:
				self.dic_lanuage_of_food[lines[0]].append(lines[1])
		#print self.dic_lanuage_of_food
		#'''
		for keys, value in self.dic_lanuage_of_food.iteritems():
			for index, v in enumerate(value):
				#print (keys, self.dic_language_desc[v])
				if keys not in self.temp_dic_lanuage_of_food:
					self.temp_dic_lanuage_of_food[keys] = []
					self.temp_dic_lanuage_of_food[keys].append(self.dic_language_desc[v])
				else:
					self.temp_dic_lanuage_of_food[keys].append(self.dic_language_desc[v])
		# for keys, value in self.temp_dic_lanuage_of_food.iteritems():
		# 	for v in value:
		# 		print keys, v
		self.save(self.temp_dic_lanuage_of_food, "langua.pickle")
		temp = self.retrieve("langua.pickle")
		#print temp

	def create_food_group(self):
		self.dic_food_group = {}
		f = open(self.folder + "/" + self.food_group, "r")
		for lines in f:
			lines = lines.strip().split('^')
			lines[0] = lines[0][1:-1]
			lines[1] = lines[1][1:-1]
			self.dic_food_group[lines[0]] = lines[1]
		#print (dic)
		#self.save(dic, "food_group.pickle");
		#dic_temp = self.retrieve("food_group.pickle")

	def save(self, variable, fileName):
		base_add = "../data/food_desc_files/"
		with open(base_add+"/"+fileName, 'wb') as f:
			pickle.dump(variable, f)

	def retrieve(self, fileName):
		base_add = "../data/food_desc_files/"
		with  open(base_add+'/'+fileName, 'rb') as f:
			return pickle.load(f)


if __name__ == '__main__':
	cl = create_links("../data/sr28asc/")
	cl.create_food_group()
	cl.create_language_desc()
	cl.create_lanuage_of_food()
	cl.create_food_desc()
	cl.create_extra_food_names()
