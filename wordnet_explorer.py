import all_food_words_from_all_solutions
from nltk.corpus import wordnet as wn


def get_hypernym_names_from_list_of_paths(list_of_paths):
    flat_list = [item for sublist in list_of_paths for item in sublist]
    synset_names_set = set()
    for synset in flat_list:
        synset_names_set.add(synset.name())
    return synset_names_set


def descendant_of_food(word):
    """

    :param word:
    :return: True if the word is a descendant of "food" in WordNet 3.0, False if not
    """

    food_synset_names = [syn.name() for syn in wn.synsets('food')]
    synset_list = wn.synsets(word)
    just_nouns = list(filter(lambda x: x.pos() == 'n', synset_list))
    for synset in just_nouns:
        path_names = get_hypernym_names_from_list_of_paths(synset.hypernym_paths())
        for food_synset_name in food_synset_names:
            # print(path_names)
            # print(food_synset_name)
            if food_synset_name in path_names:
                print(path_names)
                print(food_synset_name)
                return True

    return False

if __name__ == "__main__":
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


