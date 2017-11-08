import math

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

obs_example = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'

# lookup
def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

obs_example_trans = translate_observations_to_indices(obs_example)

print(obs_example_trans)

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

print(translate_indices_to_observations(obs_example_trans))

def translate_path_to_indices(path):
    return list(map(int, path))

def translate_indices_to_path(indices):
    return ''.join([str(i) for i in indices])

path_example = '33333333333321021021021021021021021021021021021021'

translate_path_to_indices(path_example)

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


validate_hmm(hmm_7_state)

def joint_prob(model, x, z):
    pass

x_short = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'
z_short = '33333333333321021021021021021021021021021021021021'

# Your code here ...

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)

def joint_prob_log(model, x, z):
    #Consider using numpy arrays for increased precision
    z_indices = translate_path_to_indices(z)
    x_indices = translate_observations_to_indices(x)

    #Multiplied with first emission probability as seen in Alexanders slides
    prob = model.init_probs[z_indices[0]] * model.emission_probs[z_indices[0], x_indices[0]];

    for i in range(1, len(z_indices)):
        #This is true according to Alexanders slides
        prob *= model.trans_probs[z_indices[i-1]][z_indices[i]]

    #The upper bound should not matter since
    for i in range(0, len(x_indices)):
        #Reversed ordering as seen in Alexanders slides to: z_indices, x_indices.
        #Seems right since the row is the hidden states and the columns are the observations (indexing row and the column)
        prob *= model.emission_probs[z_indices[i]][x_indices[i]]

    print(prob)

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

def compute_w(model, x):
    k = len(model.init_probs)
    n = len(x)

    w = make_table(k, n)

    # Base case: fill out w[i][0] for i = 0..k-1
    # ...

    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    # ...

def opt_path_prob(w):
    pass


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

# Test implementation
w = compute_w(hmm_7_state, translate_observations_to_indices(x_short))
opt_path_prob(w)
