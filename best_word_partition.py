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
        similarity_count = float(similarity_counter[similarity])
        entropy += similarity_count*(word_count - similarity_count)/word_count
    return entropy

#print(calc_entropy(similarity_counts, len(test_words)))


# CALC ENTROPY FOR EACH WORD
# NEED TO UTILIZE WORD MAPPING IN ORDER TO CONY CALC SIM MATRIX ONCE
def find_best_word(similarity_matrix, full_word_list, viable_indices):

    word_count = len(viable_indices)
    best_entropy = 0
    best_word_index = -1
    best_similarity_dist = Counter()

    for word_index in viable_indices:

        similarities = [similarity for index, similarity in enumerate(similarity_matrix[word_index])
                        if index in viable_indices]

        similarity_dist = Counter(similarities)
        #print(similarity_dist)

        entropy = calc_entropy(similarity_dist, word_count)

        if entropy > best_entropy:
            best_entropy = entropy
            best_word_index = word_index
            best_similarity_dist = similarity_dist

    return full_word_list[best_word_index], best_entropy, best_similarity_dist

# GIVEN A SIMILARITY MATRIX AND A SUBSET OF WORDS FROM THAT MATRIX
# RETURN THE SUBSET OF THE SIMILARITY MATRIX


def get_viable_word_indices(similarity_matrix, word, target_similarity, word_dict, viable_indices):

    word_index = word_dict[word]

    return [index for index, similarity in enumerate(similarity_matrix[word_index])
            if similarity == target_similarity and index in viable_indices]

'''
    get_viable_word_indices(test_mat, "DOG", 1, word_to_index)
    get_viable_word_indices(test_mat, "DOG", 0, word_to_index)
    get_viable_word_indices(test_mat, "DOG", 3, word_to_index)
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

word_list = [word.strip() for word in words]

# print(words)

word_count = len(word_list)

# MAP WORDS TO INDICES
word_dict = word_map(word_list)


# BUILD SIMILARITY MATRIX
similarity_matrix = calc_similarity_matrix(word_list)

if np.linalg.matrix_rank(similarity_matrix) == word_count:
    print("\nPuzzle Solveable: Full Rank Matrix\n")

viable_word_indices = list(range(len(word_list)))

while True:

    if len(viable_word_indices) == 1:
        print("\nThe password is {0}\n".format(word_list[viable_word_indices[0]]))
        break

    best_word, expected_eliminations, similarity_distribution = find_best_word(similarity_matrix, word_list,
                                                                               viable_word_indices)

    print("\nBest Word Choice : {0}\nExpected Eliminations: {1}".format(best_word, expected_eliminations))
    print("Likeliness Distribution: ")
    print(sorted(similarity_distribution.items()))
    print()

    likeness = input("Enter likeness of best word or type 'ESC' to escape: \n")

    if likeness == "ESC":
        break

    likeness = int(likeness)

    assert str(type(likeness)) == "<class 'int'>"
    assert likeness in similarity_distribution

    viable_word_indices = get_viable_word_indices(similarity_matrix, best_word, likeness, word_dict,
                                                  viable_word_indices)

with open("./data/Terminal Log.txt", "a") as file:
    file.write("--------------------- ")
    file.write(best_word)
    file.write("\n")
    for word in word_list:
        file.write(word)
        file.write("\n")
    file.close()






