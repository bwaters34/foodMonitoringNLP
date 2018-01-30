import pickle
from pprint import pprint 
from collections import defaultdict

class food_to_calorie:
	def __init__(self, foodName=None):
		self.foodNames_database = self.read_to_file("./data/food_desc_files/food_names.pickle")
		self.id_to_nutrition_file = self.read_to_file("./data/food_desc_files/food_to_nutrient.pickle")
		#self.cal_calorie(foodName)

	def cal_calorie(self, foodName):
		temp_calorie = 0.0
		temp_count = 0
		for name in self.foodNames_database:
			if name == foodName:
				print name, foodName, self.id_to_nutrition_file[self.foodNames_database[name]]
				temp_calorie += self.id_to_nutrition_file[self.foodNames_database[name]]
				temp_count += 1
		return temp_calorie/float(temp_count)

	def write_to_file(self, variableName, fileName):
		with open(fileName, 'w') as f:
			pickle.dump(variableName, f)

	def read_to_file(self, fileName):
		with open(fileName, 'r') as f:
			return pickle.load(f)

	def create_food_to_nutrition_data(self):
		dic = defaultdict(list)
		file_loc = open("./data/sr28asc/NUT_DATA.txt", 'r')
		for line in file_loc:
			line = [word[1:-1] if index<2 else word for index, word in enumerate(line.split('^'))]
			dic[line[0]].append(float(line[2]))
		dic_final = {}
		for food_id in dic:
			dic_final[food_id] = round(sum(dic[food_id]), 3)
		self.write_to_file(dic_final, "./data/food_desc_files/food_to_nutrient.pickle")
		#dic_temp = self.read_to_file("./data/food_desc/files/food_to_nutrient.pickle")
		#print dic_temp

if __name__ == '__main__':
	food_calorie = food_to_calorie("bread")
	#food_calorie.create_food_to_nutrition_data()
