from causal_inference import *
import os
import datetime

### ------------
### Main program
### ------------

if __name__ == "__main__":
    start = datetime.datetime.now().replace(microsecond=0)

    # Prepare our data structures based on a csv file.
    obs_pair_hash, obs_times_hash, alphabet, events, max_time = prep_data_pairs(os.path.join("data", "CoSeLog_Experiment_1.csv"))

    # The following line generates a hypothesis for all combinations of cause - effect excluding those where cause == effect.
    # In this example, we are interested in the genuine causes of process delay. Because the process delay is always registered at
    # ~198 hours into the case, the time window is clear and can be set to [0, 200]. In that sense, the time window becomes "meaningless".
    hyps = generate_hypotheses_for_effects(alphabet, ["Case Delayed"], 0, 200)
    print(f"{len(hyps)} hypotheses are generated.")

    # Test these hypotheses for prima facie causes only.
    relations = test_hypotheses(hyps, obs_times_hash, events)
    print(f"{len(relations['Case Delayed'])} are deemed prima facie causes for the effect")

    # Calculate all average epsilon values and write to csv.
    do_all_epsilon_averages(relations, os.path.join("output", "output-CoSeLog-experiment1.csv"), obs_times_hash)

    end = datetime.datetime.now().replace(microsecond=0)
    print(f"The total execution time was {end-start}")