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
    z_indices = translate_path_to_indices(z)
    x_indices = translate_observations_to_indices(x)

    prob = model.init_probs[z_indices[0]]

    for i in range(1, len(z_indices)):
        prob *= model.trans_probs[z_indices[i-1]][z_indices[i]]

    for i in range(0, len(x_indices)):
        prob *= model.emission_probs[x_indices[i]][z_indices[i]]

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

# Test implementation
w = compute_w(hmm_7_state, translate_observations_to_indices(x_short))
opt_path_prob(w)
