import sys
import numpy as np
from collections import Counter

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

def calc_entropy(similarity_counter, word_count):
    entropy = 0
    for similarity in similarity_counter:

        # OPTIMIZATION: DONT NEED LOG AND NORMALIZATION TO MAXIMIZE ENTROPY
        # prob = float(similarity_counter[similarity])/word_count
        # entropy -= prob * log2(prob)
        similarity_count = float(similarity_counter[similarity])
        entropy += similarity_count*(word_count - similarity_count)/word_count
    return entropy


def avg_entropy(similarity_matrix):

    word_count = len(similarity_matrix)

    # print("Word Count: {}".format(word_count))

    total_entropy = 0.0

    for word_index in range(word_count):

        similarity_dist = Counter(similarity_matrix[word_index])
        # print("Word Index: {}".format(word_index))
        # print("Word Similarities: {}".format(similarity_matrix[word_index]))
        # print("Similarity Dists: {}".format(similarity_dist))

        entropy = calc_entropy(similarity_dist, word_count)
        # print("Current Entropy: {}".format(entropy))

        total_entropy += entropy
        # print("Total Entropy: {}".format(total_entropy))
    return float(total_entropy)/word_count

with open("./data/Ideal_Partioning_Example.txt") as file:
    words = file.readlines()

words = [word.strip() for word in words]

ideal_sim_matrix = calc_similarity_matrix(words)

with open("./data/Problematic_Partioning_Example.txt") as file:
    words = file.readlines()

file.close()

words = [word.strip() for word in words]

problem_sim_matrix = calc_similarity_matrix(words)

print(ideal_sim_matrix)
print(problem_sim_matrix)

# See if average similarity is an indicator
print("Avg Similarity of Ideal Word List: {0}".format(np.mean(ideal_sim_matrix)))

print("Avg Similarity of Problematic Word List: {0}".format(np.mean(problem_sim_matrix)))

ideal_eig_w, ideal_eig_v = np.linalg.eigh(ideal_sim_matrix)

prob_eig_w, prob_eig_v = np.linalg.eigh(problem_sim_matrix)

# Problematic eigven values near zero and has an eigen value of 1.0
# One or more (degenerate) 0 eigen values means theres a nulls space
print(ideal_eig_w)
print(prob_eig_w)

print(ideal_eig_v)
print(prob_eig_v)

# MATRICES ARE SYMMETRIC
np.array_equal(np.transpose(ideal_sim_matrix), ideal_sim_matrix)
np.array_equal(np.transpose(problem_sim_matrix), problem_sim_matrix)

# PROBLEMATIC SIMILARITY MATRIX HAS COMPLEX EIGEN VECTORS/VALUES

# ANTISYMMETRIC MATRIX? : −A = AT

# IDEAL MATRIX INVERTIBLE
print(np.linalg.inv(ideal_sim_matrix))

# PROBLEMATIC MATRIX ALSO INVERTIBLE
print(np.linalg.inv(problem_sim_matrix))

# AVG ENTROPY - Problematic word set has higher average eliminations
print("Avg Eliminations of Ideal Word List: {0}".format(avg_entropy(ideal_sim_matrix)))
print("Avg Eliminations of Problematic Word List: {0}".format(avg_entropy(problem_sim_matrix)))

# Matrix Rank !!!!!!!!!!!!!!!!!!

print("Matrix Rank of Ideal Word List: {0}".format(np.linalg.matrix_rank(ideal_sim_matrix)))
# 16  full rank

print("Matrix Rank of Problematic Word List: {0}".format(np.linalg.matrix_rank(problem_sim_matrix)))
# 12 - nullspace

# Determinant
print("Matrix Determinant of Ideal Word List: {0}".format(np.linalg.det(ideal_sim_matrix)))
# 547828

print("Matrix Determinant of Problematic Word List: {0}".format(np.linalg.det(problem_sim_matrix)))
# 0 - nullspace


# FIND THE NULLS SPACE - from scipy cookbook (using singular value decomposition)
def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

# Determinant
print("Nullspace of Ideal Word List:\n {0}".format(np.round(nullspace(ideal_sim_matrix), 2)))
# EMPTY

print("Nullspace of Problematic Word List:\n {0}".format(np.round(nullspace(problem_sim_matrix), 2)))
# 16x4 - 4 columns - 4 ways to get 0

null_space = nullspace(problem_sim_matrix)

print(np.round(np.matmul(problem_sim_matrix, null_space), 2))

game_test_list = ["MORE"
                , "FORK"
                , "CAPE"
                , "NINE"
                , "LINK"
                , "HUNT"
                , "WILL"
                , "BUMP"
                , "LUSH"
                , "FAIL"
                , "TEST"
                , "AMMO"
                , "GOES"
                , "TILE"
                , "GATE"
                , "KITS"
                , "AGES"
                , "CHIP"]

game_sim_matrix = calc_similarity_matrix(game_test_list)

# FULL RANK
print("Matrix Rank of Game Word List: {0}".format(np.linalg.matrix_rank(game_sim_matrix)))
print("Length of Word List: {0}".format(len(game_test_list)))

#############################################################

with open("./data/1707121703.txt") as file:
    words = file.readlines()
    file.close()
words = [word.strip() for word in words]

game_sim_matrix = calc_similarity_matrix(words)

game_sim_matrix_rank = np.linalg.matrix_rank(game_sim_matrix)

word_count = len(words)

print("Matrix Rank of Game Word List: {0}".format(game_sim_matrix_rank))
print("Length of Word List: {0}".format(word_count))

if game_sim_matrix_rank == word_count:
    print("Puzzle Solveable: Full Rank Matrix")

