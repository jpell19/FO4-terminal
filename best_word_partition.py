import sys
import numpy as np
from collections import Counter
from math import log2

# CALC WHETHER POSSIBLE TO SOLVE: f(word_length, word_count, agg_similarity_score)
# USE DLND ENVIRONMENT - MAYBE MAKE A NEW PROJECT ENVIRONMENT?
# BUILD TESTER
# BUILD LOGGER - STORE RECORDS AND TRACE ATTEMPS

def pairwise_similarity(word_0, word_1):

    word_0_list = list(word_0.upper())
    word_1_list = list(word_1.upper())

    assert len(word_0) == len(word_1)  #

    similarity = sum([1 for i, char in enumerate(word_0_list) if char == word_1_list[i]])

    return similarity

# print(pairwise_similarity("GOD", "DOG"))

# ENSURE THAT WE ALWAYS MAP WORDS TO THE SAME INDEX LOCATIONS
# RETURN word_to_index, index_to_word dictionaries


def word_map(word_list):

    return {word: index for index, word in enumerate(word_list)}


# ONLY CALCULATE THIS ONCE SINCE THIS IS MOST COSTLY: O(n^2) time
def calc_similarity_matrix(word_list):

    word_count = len(word_list)
    similarity_mat = np.zeros((word_count, word_count))
    word_length = len(word_list[0])

    for word_0_index, word_0 in enumerate(word_list):

        similarity_mat[word_0_index][word_0_index] = word_length

        for word_1_index in range(word_0_index):

            similarity = pairwise_similarity(word_0, word_list[word_1_index])

            similarity_mat[word_0_index][word_1_index] = \
                similarity_mat[word_1_index][word_0_index] = similarity

    return similarity_mat

'''
    test_words = ["GOD", "DOG", "GOO"]
    word_to_index = word_map(test_words)
    test_mat = calc_similarity_matrix(test_words)
    similarity_counts = Counter(test_mat[0])
    print(test_words)
    print(test_mat)
'''

def calc_entropy(similarity_counter, word_count):
    entropy = 0
    for similarity in similarity_counter:

        # OPTIMIZATION: DONT NEED LOG AND NORMALIZATION TO MAXIMIZE ENTROPY
        # prob = float(similarity_counter[similarity])/word_count
        # entropy -= prob * log2(prob)
        similarity_count = similarity_counter[similarity]
        entropy += similarity_count*(word_count - similarity_count)
    return entropy

#print(calc_entropy(similarity_counts, len(test_words)))


# CALC ENTROPY FOR EACH WORD
# NEED TO UTILIZE WORD MAPPING IN ORDER TO CONY CALC SIM MATRIX ONCE
def find_best_word(similarity_matrix, word_list, viable_indices = None):

    word_count = len(word_list)
    entropies = np.zeros(word_count)

    for word_index, word in enumerate(word_list):
        similarity_dist = Counter(similarity_matrix[word_index])
        #print(similarity_dist)

        # TODO: CANT TAKE ENTIRE ROW OF SIMILARITY DIST (INCLUDED COLUMNS CHANGE WHEN SUBSETTING)
        entropies[word_index] = calc_entropy(similarity_dist, word_count)

    return word_list[np.argmax(entropies)]

# GIVEN A SIMILARITY MATRIX AND A SUBSET OF WORDS FROM THAT MATRIX
# RETURN THE SUBSET OF THE SIMILARITY MATRIX

def prune_word_list(similarity_matrix, word, target_similarity, word_dict):

    word_index = word_dict[word]

    return [index for index, similarity in enumerate(similarity_matrix[word_index]) if similarity == target_similarity]

'''
    prune_word_list(test_mat, "DOG", 1, word_to_index)
    prune_word_list(test_mat, "DOG", 0, word_to_index)
    prune_word_list(test_mat, "DOG", 3, word_to_index)
    prune_similarity_matrix(test_mat, [0,1])
'''



#print(find_best_word(test_mat, test_words))


###############################
# MAIN

# GET FILE NAME FROM COMMAND LINE
args = sys.argv

assert len(args) == 2

file_name = args[1]

# Open file and read lines into list
with open(file_name) as file:
    words = file.readlines()

words = [word.strip() for word in words]

# print(words)

# MAP WORDS TO INDICES
word_dict, index_dict = word_map(words)

# BUILD SIMILARITY MATRIX
similarity_matrix = calc_similarity_matrix(words)

# FIND BEST WORD
best_word = find_best_word(similarity_matrix, words, word_dict)

print("Best Word Choice : {0}".format(best_word))
print("Likeliness Distribution: ")
print(Counter(similarity_matrix[word_dict[best_word]]).most_common())
