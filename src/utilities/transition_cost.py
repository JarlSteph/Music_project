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
    
def transition_cost(song1, song2, w_h = 0.5, w_t = 0.5):
    """
    Calculate total cost based on results from two above functions

    Standard weights are 50/50

    Improvements:
    Think about how to weight the two. Tempo cost can easily dominate given 
    
    """
    harm_cost = harmonic_distance(song1["key"], song2["key"])
    tempo_cost = tempo_distance(song1["bpm"], song2["bpm"], normalize=False) # NOT normalized
    return w_h * harm_cost + w_t * tempo_cost