import all_food_words_from_all_solutions
from nltk.corpus import wordnet as wn
from collections import deque
from pprint import pprint
import random
import cPickle as pickle

def get_hypernym_names_from_list_of_paths(list_of_paths):
    flat_list = [item for sublist in list_of_paths for item in sublist]
    synset_names_set = set()
    for synset in flat_list:
        synset_names_set.add(synset.name())
    return synset_names_set


def string_is_descendant_of_food(word, setting = 'most_common', ):
    """
    :param word:
    :param setting: parameters = 'most_common', 'majority_vote' and 'all'
    :return: True if the word is a descendant of "food" in WordNet 3.0, False if not
    """
    food_synset_names = [syn.name() for syn in wn.synsets('food')]
    synset_list = wn.synsets(word)
    just_nouns = list(filter(lambda x: x.pos() == 'n', synset_list))
    if setting == 'most_common':
        # print('most common chosen')
        just_nouns = just_nouns[0:1] # create list of just first element
    elif setting == 'all':
        pass # don't change just_nouns
    elif setting == 'majority_vote':
        pass # not implemented yet
    else:
        raise ValueError
    for synset in just_nouns:
        if synset_is_descendant_of_food(synset):
            return True
    return False

def synset_is_descendant_of_food(synset):
    food_synset_names = [syn.name() for syn in wn.synsets('food')]
    path_names = get_hypernym_names_from_list_of_paths(synset.hypernym_paths())
    for food_synset_name in food_synset_names:
        # print(path_names)
        # print(food_synset_name)
        if food_synset_name in path_names:
            # print(path_names)
            # print(food_synset_name)
            return True
    return False

def calculate_percentage_of_solutions_in_wordnet():
    food_names_set = all_food_words_from_all_solutions.get_all_food_names_in_solutions()

    num_foods_with_wordnet_entries = 0

    names_not_in_wordnet =[]

    food_synset_names = [syn.name() for syn in wn.synsets('food')]
    food = wn
    for name in food_names_set:
        synset_list = wn.synsets(name)
        if len(synset_list) > 0:
            num_foods_with_wordnet_entries +=1
            distances = []
            food_related_to_at_least_one_sense_of_word = False
            for synset in synset_list:
                path_names = get_hypernym_names_from_list_of_paths(synset.hypernym_paths())
                for food_synset_name in food_synset_names:
                    # print(path_names)
                    # print(food_synset_name)
                    if food_synset_name in path_names:
                        food_related_to_at_least_one_sense_of_word = True
                        break # minor optimization, we don't need to continue searching
                if food_related_to_at_least_one_sense_of_word:
                    break # minor optimization
            print(name + ': ' + str(food_related_to_at_least_one_sense_of_word))

        else:
            names_not_in_wordnet.append(name)

    print(num_foods_with_wordnet_entries / float(len(food_names_set)))

def get_all_descendants_of_food_from_wordnet():
    food_synsets = wn.synsets('food')[:2] # ignore "food for thought" sense

    queue = deque(food_synsets)
    seen_before = set()
    synsets_found = []
    for syn in food_synsets:
        seen_before.add(syn)
        synsets_found.append(syn)
    while len(queue) > 0:
        elem = queue.pop()
        neighbors = elem.hyponyms()
        for neighbor in neighbors:
            if neighbor not in seen_before:
                synsets_found.append(neighbor)
                queue.append(neighbor)
                seen_before.add(neighbor)
        if len(queue) % 100 == 0:
            print(len(queue))
    print(synsets_found)
    print(len(synsets_found))
    new_food_names = set()
    for synset in synsets_found:
        wordnet_format_name = synset.name() # example: "sour_mash.n.02"
        name_without_periods = wordnet_format_name.split('.')[0]
        name = name_without_periods.replace('_', ' ')
        if string_is_descendant_of_food(name, 'most_common'):
            new_food_names.add(name)
    food_names_dict = dict.fromkeys(new_food_names, None)
    print(len(food_names_dict))
    with open('data/food_desc_files/wordnet_food_words.pickle', 'wb') as f:
        pickle.dump(food_names_dict, f)

if __name__ == "__main__":
    with open('data/food_desc_files/wordnet_food_words.pickle', 'rb') as f:
        wordnet_food_names = pickle.load(f)
    print(len(wordnet_food_names))
    food_list = wordnet_food_names.keys()
    food_list = sorted(food_list)
    for i, food in enumerate(food_list):
        print(i,food)

