import os 
import pickle 
import re 
from pprint import pprint 

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
		print temp_re.keys()
		temp_re = self.retrieve("food_group.pickle")
		print temp_re
		#print self.dic_food_group_final

	def create_lists(self, sentence):
		temp = []
		sentence = sentence.strip().split(',')
		temp.append(sentence[0])
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
		for keys, value in self.temp_dic_lanuage_of_food.iteritems():
			for v in value:
				print keys, v
		self.save(self.temp_dic_lanuage_of_food, "langua.pickle")
		#temp = self.retrieve("langua.pickle")
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