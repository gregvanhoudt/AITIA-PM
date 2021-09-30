def get_true_times(obs_times_hash, variable):
    """
    Get the timepoints where the variable is true in the form of a dictionary.

    Parameters:
        obs_times_hash: the dictionary containing all observations by variable as a tuple (case, time).
        variable: the system state to look for.
    
    Returns:
        A dictionary with all timestamps where the observation is present.
        Key = time unit, value = list of cases for which the variable was observed at the specific time unit.
    """
    results = obs_times_hash[variable]
    
    
    return(results)

def get_false_times(obs_pair_hash, variable):
    """
    Get the timepoints where the variable is false.

    Parameters:
        obs_pair_hash: the dictionary containing all observations by time unit as a tuple (case, variable).
        variable: the observation to look for.
    
    Returns:
        A list with all timestamps where the observation is not present
    """
    results = []
    for key in obs_pair_hash:
        obs = [item for item in obs_pair_hash[key] if item[1] == variable]
        if len(obs) == 0:
            results.append(key)
    
    return(results)

def get_other_causes(effect, cause, relations):
    """
    For each effect-cause combination, return a list of the other prima facie causes.

    Parameters:
        effect: the variable representing the effect (key of the relations dict).
        cause: the cause to filter out.
        relations: the dict containing all prima facie causes for the effects.
    
    Returns:
        A list containing the prima facie causes for an effect excluding the given cause.
    """
    results = [(c, p, q) for (c, p, q) in relations[effect] if c is not cause]
    return(results)

def calculate_probability_difference(effect, cause, x, r, s, obs_times_hash):
    """
    Calculates the epsilon_x value for a specific effect, cause, and x. The time window is assumed to be the same for all prima facie causes here.
    """
    c_trues = get_true_times(obs_times_hash, cause)

    e_trues = get_true_times(obs_times_hash, effect)

    x_trues = get_true_times(obs_times_hash, x)

    # P(e|c & x)
    c_and_x = get_ands(c_trues, x_trues, r, s)
    e_and_count = count_effect(e_trues, c_and_x)

    # P(e|not c & x)
    not_c_and_x = get_nots(c_trues, x_trues, r, s)
    e_not_count = count_effect(e_trues, not_c_and_x)

    # Result should be c_and_x - not_c_and_x. See the book for details ...
    if len(c_and_x) == 0 or len(not_c_and_x) == 0:
        return 0
    else:
        return(e_and_count / len(c_and_x) - e_not_count / len(not_c_and_x))

def get_ands(c_trues, x_trues, r, s):
    """
    Gets the timepoints where both c and x are true, related to the time windows of c and x and respecting the case to which they belong.
    It is assumed these time windows are the same for simplicity's sake.

    Parameters:
        c_trues: timepoints where c is true containing lists of cases observed at that point.
        x_trues: timepoints where x is true containing lists of cases observed at that point.
        r,s: the start and end times of the time window.
    
    Returns:
        List of tuples describing time window overlaps between c and x taking into account the case notion.
    """
    and_list = []
    range = s - r

    for t in c_trues:
        c_cases = c_trues[t]
        window1 = (t + r, t + s)
        x_candidates = [key for key in x_trues if key >= t - range and key <= t + range]
        for cand in x_candidates:
            x_cases = x_trues[cand]
            intersection = [c for c in c_cases if c in x_cases]
            window2 = (cand + r, cand + s)
            overlap = get_overlap(window1, window2)
            if overlap is not None and len(intersection) is not 0:
                and_list.append((overlap, intersection))
    
    return(and_list)

def get_nots(c_trues, x_trues, r, s):
    """
    Gets the timepoints where c is false yet x is true, related to the time windows of c and x and the case they belong to.
    It is assumed these time windows are the same for simplicity's sake.

    Parameters:
        c_trues: timepoints where c is true.
        x_trues: timepoints where x is true.
        r,s: the start and end times of the time window.
    
    Returns:
        List of tuples describing time windows where x is true but c is false.
    """
    not_list = []
    range = s - r

    for t in c_trues:
        c_cases = c_trues[t]
        window1 = (t + r, t + s)
        x_candidates = [key for key in x_trues if key >= t - range and key <= t + range]
        for cand in x_candidates:
            x_cases = x_trues[cand]
            intersection = [c for c in c_cases if c in x_cases]
            window2 = (cand + r, cand + s)

            if len(intersection) == 0:
                not_list.append((window2, intersection))
            
            only_x = get_only_x(window1, window2)
            if only_x is not None:
                not_list.append((only_x, intersection))
    
    return(not_list)

def count_effect(e_trues, windows):
    """
    Get the number of times where e is true in the provided time windows.

    Parameters:
        e_trues: the timepoints where e is true.
        windows: a list of windows, i.e. c_and_x and not_c_and_x.

    Returns:
        The number of times (Int) e was true in the provided time windows.
    """
    res = 0

    for (ws, we), intersection in windows:
        for e in e_trues:
            e_cases = e_trues[e]
            inter = [e_case for e_case in e_cases if e_case in intersection]
            if e >= ws and e <= we and len(inter) is not 0:
                res += 1
                break
    
    return(res)


def count_effect_deprecated(e_trues, windows):
    """
    Get the number of times where e is true in the provided time windows.

    Parameters:
        e_trues: the timepoints where e is true.
        windows: a list of windows, i.e. c_and_x and not_c_and_x.

    Returns:
        The number of times (Int) e was true in the provided time windows.
    """
    res = 0
    for e in e_trues:
        for ws, we in windows:
            if e >= ws and e <= we:
                res += 1
    
    return(res)

def get_overlap(window1, window2):
    """
    Get the overlap of two time windows.
    """
    r, s = window1
    p, q = window2

    # (r, s) must always represent the first time window
    if p < r:
        r, s = window2
        p, q = window1
    
    # if window 1 ends before window 2 starts, then there is no overlap
    if s < p:
        return(None)
    else:
        return((p, s))

def get_only_x(window_c, window_x):
    """
    Of the two time windows, return the period where only factor x is observed.
    """
    r, s = window_c
    p, q = window_x

    # if c happens before x, get the latter part starting when c ends
    if r < p:
        return((s, q))
    # when x starts first, get the first part until c starts
    elif p < r:
        return((p, r))
    # when both windows are the same, return None
    else:
        return(None)
