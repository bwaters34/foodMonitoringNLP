from collections import defaultdict
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

class EatenVsAll:
    def __init__(self):
        pass

    def open_file(self, fileName):
    	with open(fileName, 'rb') as f:
    		return pickle.load(f)

    def save_file(self, fileName, Variable):
    	with open(fileName, 'wb') as f:
    		pickle.dump(Variable, f)

    def readFile(self, fileName):
        contents = open(fileName, 'r').read().split('\n')
        foodList = []
        # print(contents)
        if "UNIQUE".lower() in contents[0].lower():
            # Yes read the file
            # print (contents)
            for lines in contents[1:]:
                if lines:
                    lines = [x.lower().strip() for x in lines.split(':')][1:]
                    # foodList.append(lines)
                    foodList += lines
            return list(set(foodList))
        return None

    def storeNames(self, food_characteristics):
        foodWords = defaultdict(list)

        #Read File
        for f in food_characteristics:
            f = "../" + f
            if os.path.exists(f):
                fileName = f.split('/')[-1]
                # Read file
                foodWordsReceived = self.readFile(f)
                if foodWordsReceived is not None:
                    foodWordsReceived.sort()
                    foodWords[fileName] = foodWordsReceived
            else:
                print("Wrong Path!!! File path doesnt exist for -> ", f)
        return foodWords

    def compareFoodEaten(self, eatenFoods, allFoods):
        x = []
        no_of_eaten_food = []
        no_of_total_food = []

        for index, fileName in enumerate(eatenFoods):
            if fileName in allFoods:
                # print("Compare -> ", index, eatenFoods[fileName] allFoods[fileName])
                x.append(index)
                no_of_eaten_food.append(len(eatenFoods[fileName]))
                no_of_total_food.append(len(allFoods[fileName]))
        print(x, no_of_eaten_food, no_of_total_food)
        return x, no_of_eaten_food, no_of_total_food, eatenFoods.keys()

    def plot_hist(self, x_, y1_, y2_, fileNames):
        # https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
        # colors = ['b','g']
        # #plots the histogram
        # fig, ax1 = plt.subplots()
        # ax1.hist([y1,y2],color=colors)
        # ax1.set_xlim(-10,10)
        # ax1.set_ylabel("Count")
        # plt.tight_layout()
        # plt.show()

        x_ = [float(_) for _ in x_]
        import matplotlib.pyplot as plt
        from matplotlib.dates import date2num
        import datetime

        x = [datetime.datetime(2011, 1, 4, 0, 0),
             datetime.datetime(2011, 1, 5, 0, 0),
             datetime.datetime(2011, 1, 6, 0, 0)]
        x = date2num(x)
        print(x, x_)
        y = [4, 9, 2]
        z=[1,2,3]
        k=[11,12,13]
        x = np.asarray(x_, dtype="float")
        y = y1
        z = y2
        ax = plt.subplot(111)
        ax.bar(x-0.2, y,width=0.2,color='b',align='center', label="Foods Eaten")
        ax.bar(x, z,width=0.2,color='g',align='center', label="All Foods Spoken")
        plt.xticks(x, fileNames)
        # ax.bar(x+0.2, k,width=0.2,color='r',align='center')
        # ax.xaxis_date()
        plt.legend()
        plt.show()

if __name__ == '__main__':
    eat = EatenVsAll()

    food_eaten_files = eat.open_file('only_eaten_files_solution.pickle')
    overall_food_names = eat.open_file('food_files_solution.pickle')

    # print("Length", overall_food_names, len(overall_food_names))
    eaten_food = eat.storeNames(food_eaten_files)

    all_food = eat.storeNames(overall_food_names)
    # print(all_food, len(all_food))
    x, y1, y2, fileNames = eat.compareFoodEaten(eaten_food, all_food)
    eat.plot_hist(x, y1, y2,fileNames)
