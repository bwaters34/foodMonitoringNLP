import solution_parser
from os import path
import os
import statistics

def get_food_names_from_file(fileName):
    solution_file_path = path.join('solutions', fileName)
    solution_set_loaded = False
    try:
        print('loading solution set')
        solution_set = solution_parser.get_solution_set_from_file(solution_file_path)
        solution_set_loaded = True
    except IOError:
        print('no solution file found for: ' + solution_file_path)
    if solution_set_loaded:
        # print('loading solution set')
        # solution_set = solution_parser.get_solution_set_from_file(solution_file_path)
        print('calculating')
        food_names_only_solution_set = solution_parser.convert_solution_set_to_set_of_food_names(fileName,
                                                                                                     solution_set)
        print("loaded!: {}".format(solution_file_path))

        return food_names_only_solution_set
    else:
        return []

def get_all_food_names_in_solutions():
    """

    :return: a set of all food names across all solutions
    """
    directory_path = 'HSLLD/HV1/MT/'
    all_food_names = set()
    for filename in os.listdir(directory_path):
        file_path = directory_path + '/' + filename
        all_food_names = all_food_names.union(get_food_names_from_file(file_path))
    return all_food_names

if __name__ == "__main__":
    directory_path = 'HSLLD/HV1/MT/'
    all_food_names = set()
    food_names_per_file = []
    for filename in os.listdir(directory_path):
        file_path = directory_path + '/' + filename
        names = get_food_names_from_file(file_path)
        all_food_names = all_food_names.union(names)
        if len(names) > 0:
            food_names_per_file.append(len(names))
    # print(all_food_names)
    # print(food_names_per_file)
    print(len(food_names_per_file))
    print(len(food_names_per_file))
    print(len(all_food_names))
    print("max: {}".format(max(food_names_per_file)))
    print("mean: {}".format(sum(food_names_per_file) / len(food_names_per_file)))
    print("min: {}".format(min(food_names_per_file)))
    print("std.dev: {}".format(statistics.stdev(food_names_per_file)))