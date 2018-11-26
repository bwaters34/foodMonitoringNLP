import pickle
from display_html_2 import *
import matplotlib.pyplot as plt
def combine_accuracies(accuracies):
    sum_true_pos = 0
    sum_false_pos = 0
    sum_false_neg = 0
    for acc in accuracies:
        sum_true_pos += acc.num_true_pos
        sum_false_pos += acc.num_false_pos
        sum_false_neg += acc.num_false_neg
    combined = Accuracy(num_true_pos=sum_true_pos, num_false_pos=sum_false_pos, num_false_neg=sum_false_neg)
    return combined


if __name__ == '__main__':
    old_transcripts = ['HSLLD/HV1/MT/admmt1.cha', 'HSLLD/HV1/MT/aimmt1.cha', 'HSLLD/HV1/MT/allmt1.cha', 'HSLLD/HV1/MT/anamt1.cha', 'HSLLD/HV1/MT/annmt1.cha', 'HSLLD/HV1/MT/aprmt1.cha', 'HSLLD/HV1/MT/bramt1.cha', 'HSLLD/HV1/MT/brimt1.cha', 'HSLLD/HV1/MT/brnmt1.cha', 'HSLLD/HV1/MT/brtmt1.cha', 'HSLLD/HV1/MT/casmt1.cha', 'HSLLD/HV1/MT/conmt1.cha', 'HSLLD/HV1/MT/davmt1.cha', 'HSLLD/HV1/MT/diamt1.cha', 'HSLLD/HV1/MT/emimt1.cha', 'HSLLD/HV1/MT/ethmt1.cha', 'HSLLD/HV1/MT/geomt1.cha', 'HSLLD/HV1/MT/gilmt1.cha', 'HSLLD/HV1/MT/gremt1.cha', 'HSLLD/HV1/MT/guymt1.cha']

    new_transcripts = """HSLLD/HV1/MT/geomt1.cha
    HSLLD/HV1/MT/kurmt1.cha
    HSLLD/HV1/MT/brnmt1.cha
    HSLLD/HV1/MT/seamt1.cha
    HSLLD/HV1/MT/maymt1.cha
    HSLLD/HV1/MT/zenmt1.cha
    HSLLD/HV1/MT/tamtp1.cha
    HSLLD/HV1/MT/megmt1.cha
    HSLLD/HV1/MT/davmt1.cha
    HSLLD/HV1/MT/vicmt1.cha
    HSLLD/HV2/MT/stnmt2.cha
    HSLLD/HV2/MT/ethmt2.cha
    HSLLD/HV2/MT/vicmt2.cha
    HSLLD/HV2/MT/megmt2.cha
    HSLLD/HV2/MT/seamt2.cha
    HSLLD/HV2/MT/petmt2.cha
    HSLLD/HV2/MT/kurmt2.cha
    HSLLD/HV2/MT/zenmt2.cha
    HSLLD/HV2/MT/karmt2.cha
    HSLLD/HV2/MT/brnmt2.cha
    HSLLD/HV2/MT/rosmt2.cha
    HSLLD/HV2/MT/diamt2.cha
    HSLLD/HV2/MT/catmt2.cha
    HSLLD/HV2/MT/inamt2.cha
    HSLLD/HV3/MT/melmt3.cha
    HSLLD/HV3/MT/ethmt3.cha
    HSLLD/HV3/MT/bobmt3.cha
    HSLLD/HV3/MT/sarmt3.cha
    HSLLD/HV3/MT/deamt3.cha
    HSLLD/HV3/MT/kurmt3.cha
    HSLLD/HV3/MT/brnmt3.cha
    HSLLD/HV3/MT/stnmt3.cha
    HSLLD/HV3/MT/tommt3.cha
    HSLLD/HV3/MT/rosmt3.cha
    HSLLD/HV3/MT/vicmt3.cha
    HSLLD/HV3/MT/devmt3.cha
    HSLLD/HV3/MT/geomt3.cha
    HSLLD/HV3/MT/catmt3.cha
    HSLLD/HV5/MT/jammt5.cha
    HSLLD/HV5/MT/rosmt5.cha
    HSLLD/HV5/MT/ethmt5.cha
    HSLLD/HV5/MT/mrkmt5.cha
    HSLLD/HV5/MT/brnmt5.cha
    HSLLD/HV5/MT/petmt5.cha
    HSLLD/HV5/MT/melmt5.cha
    HSLLD/HV5/MT/zenmt5.cha
    HSLLD/HV5/MT/kurmt5.cha
    HSLLD/HV5/MT/bramt5.cha
    HSLLD/HV5/MT/shlmt5.cha
    HSLLD/HV5/MT/davmt5.cha
    HSLLD/HV7/MT/tommt7.cha
    HSLLD/HV7/MT/admmt7.cha
    HSLLD/HV7/MT/davmt7.cha
    HSLLD/HV7/MT/allmt7.cha
    HSLLD/HV7/MT/jebmt7.cha""".splitlines(False)
    new_transcripts = [n.strip() for n in new_transcripts]

    file_paths = list(set(old_transcripts+new_transcripts))

    pickle_file_name = 'precision_and_recall_bucketed_by_hv.data'
    try:
        with open(pickle_file_name, 'rb') as f:
            precision, recall, all_results = pickle.load(f)
    except IOError:
        precision, recall, all_results = evaluate_all_files_in_directory(None,
                                                                              only_files_with_solutions=True,
                                                                              use_wordnet=False,
                                                                              use_wordnet_food_names=True,
                                                                              use_pattern_matching=True,
                                                                              use_word2vec_model = False,
                                                                              use_pretrained_Google_embeddings=True,
                                                                              file_paths=file_paths,
                                                                              remove_non_eaten_food=False,
                                                                              use_edit_distance_matching=False,
                                                                              base_accuracy_on_how_many_unique_food_items_detected=True)
        with open(pickle_file_name, 'wb') as f:
            pickle.dump((precision, recall, all_results), f)

    results_per_file = all_results.results_per_file
    results_bucketed = {}
    for i in range(1,8):
        results_bucketed[i] = []
    # filter into hv1, hv2, hv3 buckets....
    for file_name, res in results_per_file:
        home_visit = int(file_name[8])
        print(file_name, home_visit)
        results_bucketed[home_visit].append((file_name, res))
    print(results_bucketed)

    precision_recall_dict = {}

    for i in range(1,8):
        if i == 4 or i == 6:
            continue
        MT_name = str(i)
        just_accuracies = [x[1] for x in results_bucketed[i]]
        combined = combine_accuracies(just_accuracies)
        print(MT_name, combined)
        pres = calculate_precision(combined.num_true_pos, combined.num_false_pos)
        rec = calculate_recall(combined.num_true_pos, combined.num_false_neg)
        precision_recall_dict[MT_name] = (pres, rec)
    print(precision_recall_dict)


    pres_rec_list = sorted(precision_recall_dict.items())
    print(pres_rec_list)
    fileNames, pres_and_recs = zip(*pres_rec_list)
    x = range((len(fileNames)))
    y1, y2 = zip(*pres_and_recs)

    x_ = [float(_) for _ in x]
    x = np.asarray(x_, dtype="float")
    y = y1
    z = y2
    ax = plt.subplot(111)
    ax.yaxis.grid(True)
    rects1 = ax.bar(x - 0.2, y, width=0.2, color='b',
           align='center', label="Precision",)
    rects2 = ax.bar(x, z, width=0.2, color='g',
           align='center', label="Recall")

    rect_labels = []

    for rect in rects1:
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        height = int(rect.get_height())

        if height < 5:
            # Shift the text to the right side of the right edge
            yloc = height+ 1
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            yloc = 0.98 * height
            # White on magenta
            clr = 'white'
            align = 'right'

            # Center the text vertically in the bar
        xloc = rect.get_x() + rect.get_width() / 2.0
        label = ax.text(xloc, yloc, height, horizontalalignment=align,
                         verticalalignment='center', color=clr, weight='bold',
                         clip_on=True)
        rect_labels.append(label)

    print('RESULTS')
    print(pres_and_recs)
    print(fileNames)
    for key, val in results_bucketed.iteritems():
        print(key, len(val))

    plt.xticks(x, fileNames)
    plt.legend()
    plt.ylabel("Number of food keywords")
    plt.xlabel("Home visit number")
    plt.title("Home Visit vs. Precision and Recall")
    plt.show()

