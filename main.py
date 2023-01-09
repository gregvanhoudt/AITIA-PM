from Inference import Inference
from Hypothesizer import Hypothesizer
import os
import datetime

### ------------
### Main program
### ------------
data = os.path.join("Data", "Road_Traffic_Fine_Management_Process_Filtered.xes")
define_search_space = True

### ------------------------------------------------------
### Road Traffic Fine Management - 01/01/2011 - 18/06/2013
### ------------------------------------------------------

if __name__ == "__main__":
    if define_search_space:
        start = datetime.datetime.now().replace(microsecond=0)
        # Create Hypothesizer object and format the log
        hyp = Hypothesizer(data)

        hyp.prepare_event_log("hours")

        # Show the different log attributes
        print(hyp.data.columns)
        print(hyp.data.head())


        # Define the search space. Can be based on activities, resources, (case) attributes.

        ### As the problem, we consider a fine is sent for credit collection (no matter if the payment eventually arrived or not). That raw observation is therefore enough. We filtered out where a fine was immediatly payed
        hyp.observe_exists_single_value('concept:name', 'Send for Credit Collection')

        ### All cases therefore have required some form of legal steps. But not all steps are always executed. Therefore, check if the surrounding activities can be causes and also include the resources which executed them
        hyp.observe_exists(activities=True, resources=True)

        ### There is a case attribute 'Vehicle Class'. Maybe owners of a certain vehicle class are reluctant to pay?
        hyp.observe_exists(attributes={'vehicleClass'})

        ### Perhaps we sent it to credit collection because we also sent the fine late, or not at all? As such, also observe the timing of that activity more specifically. We do it two-fold:
        hyp.observe_not_exists_attribute('concept:name', 'Send Fine', 30*24)
        hyp.observe_not_exists_attribute('concept:name', 'Send Fine', 60*24)

        ### It seems that, after Insert Fine Notification, three possible paths can follow. As such, we are interested to know which path of legal action causes credit collection
        hyp.observe_directly_follows('Insert Fine Notification', 'Appeal to Judge')
        hyp.observe_directly_follows('Insert Fine Notification', 'Insert Date Appeal to Prefecture')
        hyp.observe_directly_follows('Insert Fine Notification', 'Add penalty')


        print(hyp.data.head())

        # Filter out all possible NaN values, as well as 'double' observations of Send for Credit Collection
        hyp.filter_observations_NaN()
        hyp.observations = hyp.observations[hyp.observations['observation'] != "Send for Credit Collection - "]
        print(hyp.observations)

        end = datetime.datetime.now().replace(microsecond=0)
        print(f"=== Total Hypothesiser time:\t\t {(end-start)}. ===")

        hyp.export_observations_to_csv(os.path.join("Data", "RTFM Search Space.csv"))


    start2 = datetime.datetime.now().replace(microsecond=0)

    # Create Inference object to learn root causes
    inference = Inference(os.path.join("Data", "RTFM Search Space.csv"), pb = True)
    inference.generate_hypotheses_for_effects(causes = inference.alphabet, effects = ["Send for Credit Collection"])
    inference.test_hypotheses()
    inference.calculate_average_epsilons(os.path.join("Output", "RTFM.csv"))

    end2 = datetime.datetime.now().replace(microsecond=0)
    print(f"=== Total Inference time:\t\t{end2-start2}. ===")

    if define_search_space:
        print(f"=== Total AITIA-PM time:\t\t{end2-start}. ===")