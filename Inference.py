from typing import Tuple
import pandas as pd
import tqdm

class Inference:

    def __init__(self, file_path, pb) -> None:
        self.source = pd.read_csv(file_path)
        self.pb = pb
        
        self.dict_by_time = {}
        self.dict_by_obs = {}
        self.alphabet = []
        self.events = 0
        self.max_time = 0

        self.hypotheses = []
        self.prima_facie = {}

        self.populate_vars()

    def populate_vars(self) -> None:
        """
        Read time series data from file and get *dict_by_time*, *dict_by_obs*, and *alphabet*.
        """
        self.events = len(self.source.index)

        for index, row in self.generate_iterator(self.source.iterrows(), "Processing time series data"):
            case, obs, t = row

            # Populate dict_by_time
            # Overview of all observations by timestamp
            # (In case you need to know what observations were made at a specific time)
            if t not in self.dict_by_time:
                self.dict_by_time[t] = [(case, obs)]
            else:
                self.dict_by_time[t].append((case, obs))
            
            # Populate dict_by_obs
            # Overview of all timestamps by observation
            # (In case you need to know when specific observations were made)
            if obs not in self.dict_by_obs:
                self.dict_by_obs[obs] = {t: [case]}
            else:
                if t in self.dict_by_obs[obs]:
                    self.dict_by_obs[obs][t].append(case)
                else:
                    self.dict_by_obs[obs][t] = [case]

            # Populate the alphabet
            # Overview of all different observations made (states)
            if obs not in self.alphabet:
                self.alphabet.append(str(obs))
            
            # Set max time. Required in case no window is provided in the analyses.
            self.max_time = self.source.iloc[-1,2]

    def generate_hypotheses_for_effects(self, causes, effects, window = None) -> None:
        """
        Generates hypotheses for all effects. A hypothesis is of form:
            (cause effect window-start window-end)

        Parameters:
            causes:     a variable or set of variables
            effectss:   a list of possible effects
            window:     a tuple containing the start and end of the time window.
        """
        self.hypotheses = []
        if window != None:
            window_hyps = window
        else:
            window_hyps = (0, self.max_time)
            print(f"Window was not provided. Defaulting to the entire duration of the event log: (0, {self.max_time}).")

        for effect in self.generate_iterator(effects, f"Generating hypotheses for {effects}"):
            self.hypotheses.extend(self.generate_hypotheses_for_effect(causes, effect, window_hyps))

    def generate_hypotheses_for_effect(self, causes, effect, window) -> list:
        """
        Lists all hypotheses for a given effect. Excludes hypotheses where the cause and effect are the same variable.
        
        See the docs of `generate_hypotheses_for_effects`
        """
        hyps = []
        for cause in causes:
            if cause != effect:
                hyps.append((cause, effect, window))
        return hyps

    def test_hypotheses(self) -> None:
        """
        For a hypothesis of form (c,e,(r,s)), test whether c is a potential cause of e related to time window [r, s].
        Also gets *relations*.
        """
        for hypothesis in self.generate_iterator(self.hypotheses, "Testing for prima facie conditions"):
            cause, effect, window = hypothesis

            c_and_e, c_trues, e_trues = self.test_pair_window(cause, effect, window)

            if self.is_prima_facie(c_and_e, c_trues, e_trues):
                # Add entry to Prima Facie dict containing all causes and their time windows
                if effect in self.prima_facie:
                    self.prima_facie[effect].append((cause, window))
                else:
                    self.prima_facie[effect] = [(cause, window)]

    def test_pair_window(self, cause, effect, window) -> Tuple[int, int, int]:
        """
        Get the amount of times that the cause and effect were observed together, taking into account the time window.
        In other words: for each time that the cause was observed, we count how often e was observed in the following window.

        Also take into account that the cause and effect must take place within the same case.
        """
        c_true_times = self.dict_by_obs[cause]
        e_true_times = self.dict_by_obs[effect]
        c_and_e = 0 # The amount that E was observed in a time window following C

        r, s = window

        for time in c_true_times:
            c_cases = c_true_times[time]

            e_trues_windowed = { key:val for key,val in e_true_times.items() if key >= time + r and key <= time + s }
            e_cases = []
            # The time of e is irrelevant, the dict comprehension made sure it falls in the window.
            # As such, just get the complete list of e_cases by dropping the dict structure
            for e_obs in e_trues_windowed.values():
                e_cases.extend(e_obs)
            
            # We now know how often they were observed together at the specific time.
            # Count for each case in the causes if that same case is also found in the effects list.
            for case in c_cases:
                if case in e_cases:
                    c_and_e += 1

        return(c_and_e, len(c_true_times), len(e_true_times))

    def is_prima_facie(self, c_and_e, c_trues, e_trues) -> bool:
        """
        Determines whether c is a prima facie cause of e.

        Parameters:
            c_and_e:    number of times both events were true in a time window
            c_true:     number of times the cause was true in a time window
            e_true:     number of times the effect was true in a window
        """
        if c_trues == 0:
            return(False)
        
        return (c_and_e / c_trues > e_trues / self.events)

    def calculate_average_epsilons(self, target_file) -> None:
        """
        Get the epsilon values for all relationships

        Parameters:
            target_file: the output file to write results to.
        """
        with open(target_file, mode='w') as f:
            f.write(f"cause,effect,w-start,w-end,epsilon")

            for effect in self.prima_facie:
                for cause, window in self.generate_iterator(self.prima_facie[effect], desc = "Calculating Epsilon values"):
                    eps = self.get_epsilon_average(effect, cause)
                    f.write("\n")
                    f.write(f"{cause},{effect},{window[0]},{window[1]},{eps}")

    def get_epsilon_average(self, effect, cause) -> float:
        """
        Calculates the epsilon value for a given relationship.

        Parameters:
            effect: the variable representing the effect.
            cause: the variable representing the prima facie cause
            window: the time window
        """
        other_causes = [(c, window) for (c, window) in self.prima_facie[effect] if c != cause]

        if len(other_causes) != 0:
            eps_x = 0
            for x, window in other_causes:
                # Sum epsilon_x for the other causes
                eps_x += self.calculate_probability_differences(effect, cause, x, window)

            return(eps_x / len(other_causes))
        
        return None

    def calculate_probability_differences(self, effect, cause, x, window) -> float:
        """
        Calculates the epsilon_x value for a specific effect, cause, and x. The time window is assumed to be the same for all prima facie causes here.
        """
        c_trues = self.dict_by_obs[cause]
        e_trues = self.dict_by_obs[effect]
        x_trues = self.dict_by_obs[x]

        # P(effect | cause & other cause)
        c_and_x = Inference.get_ands(c_trues, x_trues, window)
        e_and_count = Inference.count_effect(e_trues, c_and_x)

        # P(effect | not cause but other cause)
        not_c_and_x = Inference.get_nots(c_trues, x_trues, window)
        e_not_count = Inference.count_effect(e_trues, not_c_and_x)

        # Result should be c_and_x - not_c_and_x. See the book for details ...
        if len(c_and_x) == 0 or len(not_c_and_x) == 0:
            return 0
        else:
            return(e_and_count / len(c_and_x) - e_not_count / len(not_c_and_x))

    #########
    # Other #
    #########

    def generate_iterator(self, iter, desc = None):
        """
        Displays a progress bar with description if set in the initialisation of the instance.
        """
        if not self.pb:
            return iter
        else:
            return tqdm.tqdm(iter,  desc = desc)

    @staticmethod
    def get_ands(c_trues, x_trues, window) -> list:
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
        r, s = window
        range = s - r

        for t in c_trues:
            c_cases = c_trues[t]
            window1 = (t + r, t + s)
            x_candidates = [key for key in x_trues if key >= t - range and key <= t + range]
            for cand in x_candidates:
                x_cases = x_trues[cand]
                intersection = [c for c in c_cases if c in x_cases]
                window2 = (cand + r, cand + s)
                overlap = Inference.get_overlap(window1, window2)
                if overlap != None and len(intersection) != 0:
                    and_list.append((overlap, intersection))
        
        return(and_list)

    @staticmethod
    def get_nots(c_trues, x_trues, window) -> list:
        """
        Gets the timepoints where c is false yet x is true, related to the time windows of c and x and the case they belong to.
        It is assumed these time windows are the same for simplicity's sake.

        Parameters:
            c_trues: timepoints where c is true.
            x_trues: timepoints where x is true.
            window: a tuple containing the start and end times of the time window.
        """
        not_list = []
        r, s = window
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
                
                only_x = Inference.get_only_x(window1, window2)
                if only_x != None:
                    not_list.append((only_x, intersection))
        
        return(not_list)

    @staticmethod
    def count_effect(e_trues, windows) -> int:
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
                if e >= ws and e <= we and len(inter) != 0:
                    res += 1
                    break
        
        return(res)

    @staticmethod
    def get_overlap(window1, window2) -> Tuple[float, float]:
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

    @staticmethod
    def get_only_x(window_c, window_x) -> Tuple[float, float]:
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
