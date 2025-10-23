import math

def harmonic_distance(key1, key2):
    """
    Calculate the harmonic distance between two songs based on Camelot notation.

    Improvements: 
    Weight the penalties differently based on modality and wheel distance.
    (find ideal non-linearity)

    """
    def parse_camelot(key: str):
        num = int(key[:-1])
        mode = key[-1].upper()
        return num, mode

    num1, mode1 = parse_camelot(key1)
    num2, mode2 = parse_camelot(key2)

    # Distance along the 12-step Camelot circle
    wheel_distance = abs(num1 - num2)
    wheel_distance = min(wheel_distance, 12 - wheel_distance)

    # Mode difference penalty
    if mode1 == mode2:
        mode_penalty = 0
    else:
        # Parallel major/minor (e.g., 8B vs 8A)
        if num1 == num2:
            mode_penalty = 1
        else:
            # Different mode and different key
            mode_penalty = 2

    return wheel_distance + mode_penalty

def tempo_distance(bpm1, bpm2, max_diff = 30.0, normalize = True):
    """
    Calculate tempo distance between two songs based on tempo in BPM
    """
    diffs = [
        abs(bpm1 - bpm2),
        abs(bpm1 - 2 * bpm2),
        abs(2 * bpm1 - bpm2)
    ]

    min_diff = min(diffs)

    if normalize:
        # Cap at max_diff to avoid extreme values dominating
        min_diff = min(min_diff, max_diff)
        return min_diff / max_diff
    else:
        return min_diff
    

#-------- UPDATED FUNCTIONS --------

def bpm_diff(bpm1, bpm2):
    t, p = float(bpm1), float(bpm2)
    if t <= 0 or p <= 0:
        return 0.0
    d = abs(math.log2(p / t)) % 1.0   # wrap by octaves
    d = min(d, 1.0 - d)               # fold to [0, 0.5]
    return 1- (1.0 - 2.0 * d)              # score in [0,1]; 1 = perfect/octave

def key_diff(key1, key2):  
    key1 = key1.strip(); key2 = key2.strip()
    nr_t, let_t = int(key1[:-1]), key1[-1]
    nr_p, let_p = int(key2[:-1]), key2[-1]
    
    nr_diff = abs(nr_t - nr_p)
    nr_diff = min(nr_diff, 12 - nr_diff)  # circular distance
    
    let_diff = 0 if let_t == let_p else 1
    return (nr_diff + let_diff) / 7


def transition_cost(song1, song2, w_h = 0.5, w_t = 0.5):
    """
    Calculate total cost based on results from two above functions

    Standard weights are 50/50

    Improvements:
    Think about how to weight the two. Tempo cost can easily dominate given 
    
    """
    harm_cost = key_diff(song1["key"], song2["key"])
    tempo_cost = bpm_diff(song1["bpm"], song2["bpm"]) 
    return w_h * harm_cost + w_t * tempo_cost


