from Inference import Inference
from Hypothesizer import Hypothesizer
import os
import datetime
import numpy as np

### ------------
### Main program
### ------------
data = os.path.join("Data", "VSI_Revision.xes")
define_search_space = True
do_inference = True

### -----------------------------------------------------
### Artificial Data - Customer Complaint Handling Process
### -----------------------------------------------------

if __name__ == "__main__":
    if define_search_space:
        start = datetime.datetime.now().replace(microsecond=0)
        # Create Hypothesizer object and format the log
        hyp = Hypothesizer(data)

        hyp.prepare_event_log("hours", sample = 5000)

        # Show the different log attributes
        # print(hyp.data.columns)
        # print(hyp.data.head())


        # Define the search space. Can be based on activities, resources, (case) attributes.

        ### As the problem, we consider a customer is unhappy. This is registered in the data as an activity.
        hyp.observe_exists_single_value('concept:name', 'Unresolved Complaint')

        ### We expect the lack of A_3 ("2nd opinion") can lead to an unhappy customer
        hyp.observe_not_exists_attribute('concept:name', '2nd Opinion Initial Assessment')

        ### When the first communication to the customer is not done in time, the customer will bear harsh feelings towards us.
        # hyp.observe_not_exists_attribute('concept:name', 'Communicate Initial Assessment', by_time = 12)
        hyp.observe_not_exists_attribute('concept:name', 'Communicate Initial Assessment', by_time = 24)
        # hyp.observe_not_exists_attribute('concept:name', 'Communicate Initial Assessment', by_time = 48)

        ### When Analyst 4 performs "Investigation and Analysis", we expect the customer to become unhappy due to sloppiness. For confounding: check all resource & activity combinations
        hyp.observe_exists(activities=True, resources=True)

        ## When Investigation and Analysis was not performed by the same resource as Resolution and Follow Up, but only when ResFol Analyst > 4
        # hyp.observe_activity_resource_relations({'Investigation and Analysis', 'Resolution and Follow Up'})

        # print(hyp.data.head())

        # Filter out all possible NaN values, as well as 'double' observations of Unresolved Complaint
        hyp.filter_observations_NaN()
        hyp.observations = hyp.observations[~hyp.observations['observation'].str.contains('Unresolved Complaint - Clerk')]

        # Also filter out the activity "Escalation to Review Complaint" as it is ALWAYS performed straight before logging the unresolved complaint.
        hyp.observations = hyp.observations[~hyp.observations['observation'].str.contains('Escalation to Review Complaint')]

        # Filter out all genuine causes which we inserted through simulation
        # hyp.observations = hyp.observations[~hyp.observations['observation'].str.contains('Communicate Initial Assessment not observed within 24 hours')]
        # hyp.observations = hyp.observations[~hyp.observations['observation'].str.contains('2nd Opinion Initial Assessment not observed')]
        # hyp.observations = hyp.observations[~hyp.observations['observation'].str.contains('Investigation and Analysis by Analyst 1 - Resolution and Follow Up by Analyst 10')]
        # hyp.observations = hyp.observations[~hyp.observations['observation'].str.contains('Investigation and Analysis by Analyst 1 - Resolution and Follow Up by Analyst 9')]
        # hyp.observations = hyp.observations[~hyp.observations['observation'].str.contains('Investigation and Analysis by Analyst 1 - Resolution and Follow Up by Analyst 10')]
        # hyp.observations = hyp.observations[~hyp.observations['observation'].str.contains('Investigation and Analysis by Analyst 2 - Resolution and Follow Up by Analyst 9')]
        # hyp.observations = hyp.observations[~hyp.observations['observation'].str.contains('Investigation and Analysis by Analyst 2 - Resolution and Follow Up by Analyst 10')]

        # Generate all - Filter out a random number of hypotheses
        hyps_to_keep = hyp.observations['observation'].unique()
        hyps_to_keep = np.random.choice(hyps_to_keep, size = 30, replace=False)
        hyps_to_keep = np.append(hyps_to_keep, "Unresolved Complaint")
        hyp.observations = hyp.observations[hyp.observations['observation'].isin(hyps_to_keep)]


        end = datetime.datetime.now().replace(microsecond=0)
        print(f"=== Total Hypothesiser time:\t\t {(end-start)}. ===")

        hyp.export_observations_to_csv(os.path.join("Data", "VSI_Revision_SearchSpace.csv"))


    if do_inference:
        start2 = datetime.datetime.now().replace(microsecond=0)

        # Create Inference object to learn root causes
        inference = Inference(os.path.join("Data", "VSI_Revision_SearchSpace.csv"), pb = True)
        inference.generate_hypotheses_for_effects(causes = inference.alphabet, effects = ["Unresolved Complaint"])
        inference.test_for_prima_facie()
        # print(inference.prima_facie)
        inference.calculate_average_epsilons(os.path.join("Output", "VSI_Revision_NoGenuines.csv"))

        end2 = datetime.datetime.now().replace(microsecond=0)
        print(f"=== Total Inference time:\t\t{end2-start2}. ===")

        if define_search_space:
            print(f"=== Total AITIA-PM time:\t\t{end2-start}. ===")