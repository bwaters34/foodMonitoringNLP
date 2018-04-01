import nltk
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
import wordnet_explorer
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import os
from itertools import tee, izip
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def flatten_list(lst):
    flattened = [item for sublist in lst for item in sublist]
    return flattened

def get_window_features_from_text_file(content, window_size):
    num_elements_on_either_side_window = (window_size - 1)

WINDOW_SIZE = 5
NUM_ELEMENTS_ON_EITHER_SIDE_WINDOW = (WINDOW_SIZE - 1) / 2 # if Window size is 5, then there are 2 elements on either side of the middle elem.
all_food_indices = set([1908, 10948, 12002, 14318, 14733, 15158, 15217, 20532, 20822, 20880, 20882, 20893, 23509, 24031, 24339, 24387, 24990, 35050, 35116, 37315, 37525, 37603, 37814, 38264, 38274, 38286, 38288, 38290, 38300, 38305, 38335, 38356, 38361, 38372, 38390, 38405, 38416, 38422, 38450, 38456, 38511, 38534, 38557, 38634, 38678, 38708, 38728, 38737, 38789, 38796, 38833, 38835, 38852, 38856, 38862, 38864, 38922, 38964, 38989, 50216, 56504, 56505, 70292, 71383, 80708, 81263, 81280, 81282, 81288, 82740, 82758, 82763, 82765, 85323, 85477, 85524, 85812, 85828, 85844, 85874, 85877, 85895, 85896, 86046, 86390, 93971, 95011, 95048, 95069, 95150, 95159, 95168, 95236, 96425, 96813, 97064, 102446, 122817, 123759, 123779, 123794, 125713, 125715, 125716, 125722, 128923, 132033, 132109, 136777, 136792, 137908, 138019, 138023, 138031, 138055, 138061, 138159, 138340, 138411, 138583, 138602, 138604, 138606, 138608, 142433, 142925, 143083, 143433, 143439, 143976, 144281, 145622, 146889, 147905, 148733, 153857, 154092, 154096, 155589, 156226, 160135, 160177, 160606, 160630, 160634, 160638, 160639, 160643, 160647, 160651, 160945, 160992, 161019, 161241, 161472, 161869, 161885, 162593, 162657, 162673, 162690, 162705, 162732, 162788, 162797, 162803, 162807, 162855, 163407, 163410, 163482, 163608, 163610, 163615, 163781, 163969, 164030, 164309, 164312, 164352, 164426, 164563, 165290, 165292, 165294, 165296, 165314, 165358, 165360, 165433, 165501, 165520, 165532, 165586, 165749, 165752, 165756, 165803, 166024, 166737, 166965, 166979, 166984, 166990, 167000, 167010, 168195, 169328, 169435, 170106, 170397, 170462, 170474, 170807, 171439, 171805, 171812, 172461, 172721, 173104, 173327, 176214, 176220, 176235, 176421, 177189, 177332, 177769, 177884, 177908, 178167, 178330, 180062, 180064, 181535, 181851, 181858, 181871, 181899, 181911, 182220, 182374, 182398, 185712, 186313, 186361, 186382, 186441, 186487, 187053, 187055, 187395, 187493, 187923, 188237, 188244, 188424, 188429, 188603, 188660, 188693, 188765, 190118, 190215, 190442, 191101, 191106, 191153, 191177, 191224, 191271, 191296, 191368, 191417, 191443, 192267, 192764, 192768, 192771, 193462, 193711, 194247, 194411, 194644, 194647, 194840, 194895, 194900, 194936, 195049, 195051, 195118, 195236, 195354, 195591, 196413, 196463, 196543, 196603, 197345, 197896, 197947, 199272, 199273, 199282, 199388, 199393, 200257, 200599, 201269, 201522, 202451, 203025, 203247, 204134, 204258, 204986, 205542, 206509, 208902, 208923, 213361, 215529, 215957, 216446, 216917, 217290, 217340, 217406, 219060, 219079, 219093, 219125, 219133, 219263, 219404, 219626, 220053, 220071, 221992, 222949, 223511, 223876, 223921, 223970, 223988, 224104, 224135, 224164, 224199, 224220, 224355, 224361, 224370, 224401, 224500, 224502, 224618, 224633, 226161, 226207, 226213, 226249, 227383, 228180, 228461, 228856, 229073, 238149, 238229, 238246, 238263, 238275, 238315, 238324, 238362, 238382, 238401, 238413, 238435, 238609, 238643, 238661, 238837, 238912, 238953, 238957, 238983, 238992, 238999, 239008, 239022, 239026, 239034, 239054, 239058, 239067, 239075, 239084, 239101, 239113, 239166, 239179, 239226, 239258, 239301, 239352, 239437, 239602, 239630, 239655, 239744, 239885, 239941, 240796, 242029, 242031, 242725, 242768, 243832, 243862, 243893, 243896, 243911, 243923, 243926, 244222, 244252, 246152, 246199, 246213, 246467, 248029, 249430, 249593, 249620, 249634, 250351, 250354, 250849, 254370, 255345, 256339, 257990, 258466, 258471, 258474, 258536, 258541, 258821, 258823, 258825, 259505, 259536, 259712, 260355, 260357, 260973, 261650, 261687, 261878, 262008, 263363, 263529, 263547, 263619, 263690, 263816, 263935, 263972, 264011, 267143, 271864, 272019, 272355, 272359, 272514, 272603, 272607, 273428, 281232, 282167, 282409, 282660, 283242, 283299, 283301, 283305, 283335, 283393, 283396, 286316, 291348, 291978, 300290, 306716, 306717, 306738, 306805, 324837, 328396, 335851, 339163, 347174, 360569, 360626, 360640, 360983, 360986, 361203, 361214, 361341, 361385, 361721, 362021, 362092, 362097, 362142, 362160, 362189, 362212, 362291, 362293, 362295, 362299, 362360, 362362, 362364, 362589, 363022, 363077, 363098, 363160, 363165, 364489, 364598, 364621, 365204, 365822, 367510, 367521, 367540, 367573, 367761, 367782, 370116, 370118, 370656, 370894, 372559, 372882, 373105, 373151, 374813, 374823, 375481, 375485, 377695, 377968, 378949, 381290, 381802, 381896, 381917, 381922, 382819, 383119, 383301, 385125, 385369, 387002, 387436, 388088, 390391, 394087, 394168, 394505, 394730, 394750, 395091, 395130, 395813, 395983, 398506, 399172, 399282, 400361, 400363, 400375, 400382, 400384, 400390, 400392, 400403, 400407, 400424, 400441, 400444, 403636, 403700, 403709, 405645, 408368, 408842, 408898, 409147, 409154, 409157, 409992])

# NOTE: len(semcor.tagged_chunks()) = 778587,
chunks = semcor.tagged_chunks()[:100]
print(chunks)
leaves = [chunk.leaves() for chunk in chunks]
# chunks = ['a','b','c', 'd', 'e']
length_of_leaves = len(leaves)


windows = []
labels = []
for i in xrange(length_of_leaves):
    elem = leaves[i]
    if i in all_food_indices:
        label = 1
    else:
        label = 0
    labels.append(label)

    # get NUM_ELEMENTS_ON_EITHER_SIDE strings from the left of the target string.
    # chunks contains lists, where each list contains at least 1 string (sometimes more).
    # also, if we are at the first couple chunks, we need to add _START_ as dummy variables.
    elems_to_left_unflattened = leaves[max(i-NUM_ELEMENTS_ON_EITHER_SIDE_WINDOW, 0):i]
    elems_to_left_flattened = flatten_list(elems_to_left_unflattened)
    elems_to_left = elems_to_left_flattened[-NUM_ELEMENTS_ON_EITHER_SIDE_WINDOW:] # may have extra elements, so lets keep the ones closest to our string
    while len(elems_to_left) < NUM_ELEMENTS_ON_EITHER_SIDE_WINDOW: # could have not enough elements
        elems_to_left.insert(0, "_START_")
    # get NUM_ELEMENTS_ON_EITHER_SIDE_STRINGS from the right of the target string.
    elems_to_right_unflattened = leaves[i+1:i+NUM_ELEMENTS_ON_EITHER_SIDE_WINDOW+1]
    elems_to_right_flattened = flatten_list(elems_to_right_unflattened)
    elems_to_right = elems_to_right_flattened[:NUM_ELEMENTS_ON_EITHER_SIDE_WINDOW]
    while len(elems_to_right) < NUM_ELEMENTS_ON_EITHER_SIDE_WINDOW:
        elems_to_right.append("_END_")
    # Note we skip the middle element! we don't care about it!
    # print(elems_to_left)
    # print(elems_to_right)
    # print('')
    window = elems_to_left + elems_to_right
    windows.append(window)

print('windows:')
print(windows)
for win in windows:
    assert len(win) == 2 * NUM_ELEMENTS_ON_EITHER_SIDE_WINDOW


docs = [' '.join(sublist) for sublist in windows]

get_vocab_for_train = CountVectorizer(input='content') # Using CountVectorizer as a way to get the vocabulary.
get_vocab_for_train.fit(docs)
train = get_vocab_for_train.vocabulary_

# have to get vocabulary from HSLLD, otherwise when we do testing, the size of the feature matrix will be different (as HSLLD will have different words).



# get list of filenames
dir_root = 'HSLLD/HV1/MT/'
filenames_list = []
for filename in os.listdir(dir_root):
    file_path = dir_root + '/' + filename
    filenames_list.append(file_path)

#Create colocational features
for filename in filenames_list:
    with open(filename) as f:
        list_of_strings = f.read().split()




get_vocab_for_test = CountVectorizer(input='filename') # Using CountVectorizer as a way to get the vocabulary.
get_vocab_for_test.fit(filenames_list)
test_vocab = get_vocab_for_test.vocabulary_

get_vocab_for_train = CountVectorizer() # Using CountVectorizer as a way to get the vocabulary.
get_vocab_for_train.fit(docs)
train_vocab = get_vocab_for_train.vocabulary_

print(len(train_vocab))
print(len(test_vocab))
test_vocab.update(train_vocab) # combine the two vocabularies

print(train_vocab)
final_vocab = list(test_vocab.keys())

train_cv = CountVectorizer(vocabulary=final_vocab)
train_feature_matrix = train_cv.fit_transform(docs)
print(train_feature_matrix)
print(train_cv.vocabulary_)
print(len(train_cv.vocabulary_))


# THIS DOESN'T WORK! FEATURES AREN'T COLLOCATIONAL!!!!
# test_cv = CountVectorizer(vocabulary=final_vocab, input='filename')
# test_feature_matrix = test_cv.fit_transform(filenames_list)
# print(train_feature_matrix)
# print(train_cv.vocabulary_)
# print(len(train_cv.vocabulary_))

# print('got')
# just_food = [chunks[i].label().name() for i in all_food_indices]

logreg = LogisticRegression()
logreg.fit(train_feature_matrix, labels)
logreg.predict(test_feature_matrix)

ci
