import math
import numpy as np
import pandas as pd

# representation

init_probs_7_state = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]

trans_probs_7_state = [
    [0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00],
]

emission_probs_7_state = [
    #   A     C     G     T
    [0.30, 0.25, 0.25, 0.20],
    [0.20, 0.35, 0.15, 0.30],
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
    [0.30, 0.20, 0.30, 0.20],
    [0.15, 0.30, 0.20, 0.35],
]

init_probs_31_states = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

trans_probs_31_states = [
    [0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.2, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.25, 0, 0, 0.25, 0, 0, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

emission_probs_31_states = [
    #   A     C     G     T
    [0.25, 0.25, 0.25, 0.25],
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
    [0.00, 0.00, 1.00, 0.00],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.00, 0.00, 0.00, 1.00],
    [1.00, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
    [0.00, 0.00, 1.00, 0.00],
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
    [0.00, 0.00, 0.00, 1.00],
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
    [0.00, 1.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, 0.00],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.00, 1.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
]

# Translate between OBSERVATIONS and indices
def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

# Translate between PATH and indices
def translate_path_to_indices(path):
    return list(map(int, path))

def translate_indices_to_path(indices):
    return ''.join([str(i) for i in indices])

# Translate between STATES and indices
def translate_states_to_indices(states):
    mapping = {'n': 0, 'c': 1, 'r': 2}
    return [mapping[symbol.lower()] for symbol in states]

#Used with the 31 state model
#def translate_states_to_indices(states):
    #The end values are not includes
    #mapping = {'n': 0, 'c': np.arange(1, 16), 'r': np.arange(16, 31)}
    #return [mapping[symbol.lower()] for symbol in states]


def translate_states_to_indices(states):
    def index(state):
        if state == 0: return 'n'
        if state in range(1, 16): return 'c'
        if state in range(16, 31): return 'r'

    return [index(x) for x in states]


#def translate_indices_to_path(indices):
    #mapping = {0: 'n', np.arange(1, 16): 'c', np.arange(16, 31): 'r'}


path_example = '33333333333321021021021021021021021021021021021021'

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

# Collect the matrices in a class.
hmm_7_state = hmm(init_probs_7_state, trans_probs_7_state, emission_probs_7_state)

# We can now reach the different matrices by their names. E.g.:
hmm_7_state.trans_probs

init_probs_3_state = [0.10, 0.80, 0.10]

trans_probs_3_state = [
    [0.90, 0.10, 0.00],
    [0.05, 0.99, 0.05],
    [0.00, 0.10, 0.90],
]

emission_probs_3_state = [
    #   A     C     G     T
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
]

hmm_3_state = hmm(init_probs_3_state, trans_probs_3_state, emission_probs_3_state)

#TODO: Consider using (numpy) filters for efficiency
def validate_hmm(model):
    if math.fsum(model.init_probs) != 1: return False

    for num in model.init_probs:
        if num < 0 or num > 1: return False

    for row in model.trans_probs:
        if math.fsum(row) != 1: return False
        for num in row:
            if num < 0 or num > 1: return False

    for row in model.emission_probs:
        if (math.fsum(row) != 1): return False
        for num in row:
            if num < 0 or num > 1: return False

    return True

def joint_prob(model, x, z):
    """
    Only for debuggin purposes; should not be used in practice since it is numerically unstable.
    Use joint_prob_log to make perform the same computation in 'log space'.
    """
    #Consider using numpy arrays for increased precision
    z_indices = translate_path_to_indices(z)
    x_indices = translate_observations_to_indices(x)

    #Multiplied with first emission probability as seen in Alexanders slides
    prob = model.init_probs[z_indices[0]]

    for i in range(1, len(z_indices)):
        #This is true according to Alexanders slides
        prob *= model.trans_probs[z_indices[i-1]][z_indices[i]]

    #The upper bound should not matter since
    for i in range(0, len(x_indices)):
        #Reversed ordering as seen in Alexanders slides to: z_indices, x_indices.
        #Seems right since the row is the hidden states and the columns are the observations (indexing row and the column)
        prob *= model.emission_probs[z_indices[i]][x_indices[i]]

    print(prob)
    return prob

x_short = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'
z_short = '33333333333321021021021021021021021021021021021021'

x_long = 'TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCT'

z_long = '33333210210210210210210210210210210210210210210210210210210210210210210333333333345645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645632102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102103210210210210210210210210333345645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645645633333334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563321021021021021021021021021021021021021021021021021021021021021021021021021021021021021021021021032102102102102102102102102102102102102102102102102102102102102'

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)

def joint_prob_log(model, x, z):
    #Consider using numpy arrays for increased precision
    z_indices = translate_path_to_indices(z)
    x_indices = translate_observations_to_indices(x)

    #Multiplied with first emission probability as seen in Alexanders slides
    prob = log(model.init_probs[z_indices[0]])

    for i in range(1, len(z_indices)):
        #This is true according to Alexanders slides
        prob += log(model.trans_probs[z_indices[i-1]][z_indices[i]])

    #The upper bound should not matter since
    for i in range(0, len(x_indices)):
        #Reversed ordering as seen in Alexanders slides to: z_indices, x_indices.
        #Seems right since the row is the hidden states and the columns are the observations (indexing row and the column)
        prob += log(model.emission_probs[z_indices[i]][x_indices[i]])

    return prob

# Your code here ...

for i in range(0, len(x_long), 100):
    x = x_long[:i]
    z = z_long[:i]

    x_trans = translate_observations_to_indices(x)
    z_trans = translate_path_to_indices(z)

    # Make your experiment here...

def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]

#Assume x is already indiced
def compute_w(model, x):
    k = len(model.init_probs)
    n = len(x)

    #x_indices = translate_observations_to_indices(x) # I am not sure we need this

    w = make_table(k, n)

    # Base case: fill out w[i][0] for i = 0..k-1
    for i in range(0, k):
        #Perhaps we need to use x_indices and perhaps we need to use i
        #z1 = p(z1) + log p(z1 | x1)
        #z2 = p(z2) + log p(z2 | x1)
        #z3 = p(z3) + log p(z3 | x1)
        #z4 = p(z4) + log p(z4 | x1)

        w[i][0] = log(model.init_probs[i]) + log(model.emission_probs[i][x[0]])

    # Inductive case: fill out w[j][i] for i = 0..n-1, j = 0..k-1
    for nn in range(1, n):
        # Find max in column
        previous_state = 0
        for kk in range(0, k):
            w[kk][nn] = float("-inf")
            for j in range(0, k):
               # print(f'n = {n}, k = {kk}, j = {j}, x(n) = {x[n]}')
                p1 = log(model.emission_probs[kk][x[nn]])
                p2 =  w[j][nn-1]
                p3  = log(model.trans_probs[j][kk])
                w[kk][nn] = max(w[kk][nn], p1+ p2 + p3)
        #print(f'collumn {nn} has sum {w[0][nn]+w[1][nn]+w[2][nn]} w0 = {w[0][nn]} w1 = {w[1][nn]} w2 = {w[2][nn]}')

    return w




    #        if(max(max_w, w[j][i - 1]) > max_w):
     #           previous_state = j
#
                #This has already been logged
    #            max_w = w[j][i - 1]
#
 #       for j in range(0, k):
  #          w[j][i] = max_w + log(model.trans_probs[previous_state][j]) + log(model.emission_probs[j][x[i]])

  #  return w

def backtrack_log(w, x, model):
    n = len(x)
    k = len(w)

    z = np.ones((n,), dtype=np.int) #Make a table with 1 row and N columns - one for each hidden state

    #Find a better way to get the last column
    z[n - 1] = np.argmax(np.take(w, -1, axis=1))

    #Make a backwards loop since we are backtracking
    #i corresponds to n on the slides and j to k (consider a name change)
    for i in range((n - 2), -1, -1):
        #This could have been done much more efficient with numpy arrays
        # zn = float("-inf")
        candidates = np.array(k)
        for j in range(0, k):
            #I am not sure about the last term (z[n+1])

            firstPart = log(model.emission_probs[z[i + 1]][x[i + 1]])
            #print(f"Loop with i value: {i} and j value: {j}")
            #print(firstPart)

            secondPart = w[j][i]
            #print(secondPart)

            thirdPart = log(model.trans_probs[j][z[i + 1]])
            #print(thirdPart)

            zn = firstPart + secondPart + thirdPart
            #print(zn)

            np.append(candidates, zn)

        z[i] = np.argmax(candidates)
        #print(f"Printing z[{i}] which is {z[i]}")

    return z

#This might be wrong. See the pseudo code on slides for lecture 3
def opt_path_prob(w):
    max_w = float("-inf")
    for j in range(0, len(w)):
        max_w = max(max_w, w[j][len(w[0]) - 1])

    return max_w

def count_transitions_and_emissions(K, D, x, z):
    """
    Returns a KxK matrix and a KxD matrix containing counts cf. above
    """
    # we assume that we have seen each transition once, and each observation once
    counted_trans = np.ones(shape=(K, K))
    counted_emissions = np.ones(shape=(K, D))
    x, z = np.array(x), np.array(z)
    for i in range(len(z)-1):
        counted_trans[z[i]][z[i+1]] += 1
    for i in range(len(x)):
        counted_emissions[z[i]][x[i]] += 1
    return counted_trans, counted_emissions

def training_by_counting(K, D, x, z):
    counted_trans, counted_emissions = count_transitions_and_emissions(K, D, x, z)

    trained_trans_probs = (counted_trans.T / np.sum(counted_trans, axis=1)).T
    trained_emission_probs = (counted_emissions.T / np.sum(counted_emissions, axis=1)).T

    trained_init_probs = np.ones(K)
    trained_init_probs[z[0]] += 1

    trained_init_probs = (trained_init_probs / np.sum(trained_init_probs))

    model = hmm(trained_init_probs, trained_trans_probs, trained_emission_probs)

    return model


#Notes for log transform:
#Use the rule that the logarithm of a product is the same as the logarithm of the first term plus the logarithm of the second term

#Notes for Viterbi decoding:
# 1: Given a model, find a Z that maximizes the probability of Z given X (the observations) and the model
# 2: Is calculated based on the recursive which we have seen in the lecture (and dynamic programming)
# 3: Base case p(z1) * p(x1 | z1) (this is the "bottom") of the pyramid. It corresponds to the first column in the (emission probability) table
    # NB!!! DO THAT FOR EACH Z
# 4: Now, for each state, we multiply the probability that we emit the given symbol in our state given all the previous transistions
    # Again, do that for each Z and multiply that by the maximum value of the previous state multiplied by the transistion probability from that state to our state to our state
    # For this implementation, see slides for D 27/10 slide 5 (https://blackboard.au.dk/bbcswebdav/pid-1060304-dt-content-rid-3159111_1/courses/BB-Cou-UUVA-67943/HMM-Slides/ml-3-hmm-implementations.pdf)

#Notes for scope:
#Ignore posterior decoding in the practical exercises for week 9
#Ignore everything past "training by counting" in the practical exercises for week 10 (we need training by counting, however)

def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

if __name__ == "__main__":
    obs_dict = read_fasta_file('data/genome1.fa')
    obs = obs_dict[list(obs_dict.keys())[0]]
    obs_indices = translate_observations_to_indices(obs)

    states_dict = read_fasta_file('data/true-ann1.fa')
    states = states_dict[list(states_dict.keys())[0]]
    states_indices = np.array(translate_states_to_indices(states))


    #print((states_indices == 0).sum())
    #print((states_indices == 1).sum())
    #print((states_indices == 2).sum())


    hmm_31_state_model = hmm(init_probs_31_states, trans_probs_31_states, emission_probs_31_states)
    w = compute_w(hmm_31_state_model, translate_observations_to_indices(x_short))
    print(w)
    #print(np.array(w).T[:240])
    #print(np.array(w).T[:-20])
    backtracked = backtrack_log(w, translate_observations_to_indices(x_short), hmm_31_state_model)

    #Do this better
    print((backtracked == 0).sum())
    print((backtracked == 1).sum())
    print((backtracked == 2).sum())
    print((backtracked == 3).sum())
    print((backtracked == 4).sum())
    print((backtracked == 5).sum())
    print((backtracked == 6).sum())
    print((backtracked == 7).sum())
    print((backtracked == 8).sum())
    print((backtracked == 9).sum())
    print((backtracked == 10).sum())
    print((backtracked == 11).sum())
    print((backtracked == 12).sum())
    print((backtracked == 13).sum())
    print((backtracked == 14).sum())
    print((backtracked == 15).sum())
    print((backtracked == 16).sum())
    print((backtracked == 17).sum())
    print((backtracked == 18).sum())
    print((backtracked == 19).sum())
    print((backtracked == 20).sum())
    print((backtracked == 21).sum())
    print((backtracked == 22).sum())
    print((backtracked == 23).sum())
    print((backtracked == 24).sum())
    print((backtracked == 25).sum())
    print((backtracked == 26).sum())
    print((backtracked == 27).sum())
    print((backtracked == 28).sum())
    print((backtracked == 29).sum())
    print((backtracked == 30).sum())
    print((backtracked == 31).sum())


    f= open('testoutput.fsa','w+')
    np.savetxt('testoutput.fsa', backtracked)
    f.close

    f= open('w.fsa','w+')
    np.savetxt('w.fsa', w)
    f.close

    #df = pd.DataFrame(translate_indices_to_path(backtracked), columns=['states'])
    #df.to_csv("Test.csv")

    #trained_model = hmm()
