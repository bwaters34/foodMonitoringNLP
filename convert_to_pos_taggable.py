import sys
import subprocess
import os
import CMUTweetTagger
import pickle

def create_readable_file(file_path):
    """

    :param file_path:
    :return: tuple of (String of modified file, list of line numbers from the modified file)
    """
    modified_file = []
    line_numbers = []
    with open(file_path) as f:
        line_number = 1
        for line in f:
            if line[0] == "*":
                just_words = line.split()[1:]
                modified_line = ' '.join(just_words)
                # print(str(line_number) + '\t' + modified_line)
                modified_file.append(modified_line)
                line_numbers.append(line_number)
            line_number += 1
        return modified_file, line_numbers

def create_pos_tags_pickle(save_pos_tags_pickle):
    converted_file, line_numbers = create_readable_file(save_pos_tags_pickle)
    pos_tag_tuples =  CMUTweetTagger.runtagger_parse(converted_file)
    assert len(pos_tag_tuples) == len(line_numbers)
    pos_tags_dict = dict(zip(line_numbers, pos_tag_tuples))
    file_to_save_to = "pos_tags/" + save_pos_tags_pickle
    with open(file_to_save_to, 'wb') as f:
        pickle.dump(pos_tags_dict, f)

if __name__ == "__main__":

    # Use this code to create the tags for a single file
    # if len(sys.argv) > 1: # if a file was passed in through command line
    #     file_path = sys.argv[1]
    # else:
    #     file_path = 'HSLLD/HV1/MT/admmt1.cha'
    # create_pos_tags_pickle(file_path)
    #

    # use this code to create tags for every file
    directory_path = 'HSLLD/HV1/MT/'

    for filename in os.listdir(directory_path):
        file_path = directory_path + '/' + filename
        create_pos_tags_pickle(file_path)
        print('file completed')