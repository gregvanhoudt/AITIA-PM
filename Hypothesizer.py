import os
import copy
from typing import Literal

import pandas as pd
import numpy as np
from tqdm import tqdm

import pm4py
import pm4py.util.constants as constants

class Hypothesizer:
    class_var = True
    def __init__(self, filepath: str) -> None:
        self.activities: bool = False
        self.resources: bool = False
        self.idling: bool = False
        self.caseattribute: str = False

        self.combinefactors: bool = False
        
        self.filepath = filepath
        self.data: pd.DataFrame = None
        self.data_prepped: bool = False
        self.time_unit = None
        self.observations = pd.DataFrame(columns=["case:concept:name", "observation", "time:timestamp"])

    def prepare_event_log(self, time_unit: Literal['seconds', 'minutes', 'hours']):
        # Only XES or CSV is accepted
        accepted_ext = {'.xes', '.xes.gz', '.csv'}
        if ".xes.gz" in self.filepath:
            ext = '.xes.gz'
        else:
            filename, ext = os.path.splitext(os.path.basename(self.filepath))
            del filename
        data = None

        if str.lower(ext) not in accepted_ext:
            raise ValueError(f"The algorithm only accepts files with {accepted_ext} extensions. You passed a {ext} file.")

        VALID_TIME_UNITS = {'seconds', 'minutes', 'hours'}
        if str.lower(time_unit) not in VALID_TIME_UNITS:
            raise ValueError(f"prepare_event_log: time_unit must be one of {VALID_TIME_UNITS}")

        # Define factor for time unit conversion
        time_factor = 1 if time_unit == 'seconds' else 60 if time_unit == 'minutes' else 3600
        self.time_unit = str.lower(time_unit)

        if str.lower(ext) == ".csv":
            data = pd.read_csv(self.filepath)
            # We assume names are already correct and the time is expressed in units starting at 0.


        else:
            # When the file is an XES event log, we need to convert it to a compatible Data Frame
            data = pm4py.read_xes(self.filepath)
            data = pm4py.convert_to_dataframe(data)
            data = data.sort_values('time:timestamp', ascending=True)

            # Calculate the time units (in minutes)
            data['time:timestamp'] = data['time:timestamp'].apply(lambda x: x.timestamp() / time_factor)
            mintime = data['time:timestamp'].min()
            data['time:timestamp'] = data['time:timestamp'].apply(lambda x: x - mintime)

        self.data = data.reset_index(drop=True)
        self.data_prepped = True
        print("Event log loaded.")

    def observe_exists(self, activities: bool = False, resources: bool = False, attributes: set = None):
        if self.data_prepped == False:
            raise RuntimeError(f"Before the search space can be defined, one must call the 'prepare_event_log()' function.")
        
        self.activities = activities
        self.resources = resources
        self.attributes = attributes

        search_attributes = {'concept:name' : self.activities, 'org:resource' : self.resources}
        if type(attributes) == set:
            caseattributes = [attr for attr in attributes if attr in self.data.columns]
            for attr in caseattributes:
                search_attributes[attr] = True
        if type(attributes) == str and attributes in self.data.columns:
            search_attributes[attributes] = True

        search_attributes = {k : v for k,v in search_attributes.items() if v == True}

        data = copy.deepcopy(self.data)
        # Add data column for the observations, based on the attributes set above
        for col, val in tqdm(search_attributes.items(), desc = f"Observe EXISTS - {search_attributes}"):
            if val and col != None:
                if 'observation' not in data.columns:
                    data = data.assign(observation = self.data[col])
                else:
                    data['observation'] = data[['observation', col]].agg(' - '.join, axis = 1)

        self.observations = self.observations.append(data[["case:concept:name", "observation", "time:timestamp"]], ignore_index=True)

        self.arrange_observations()
        # print("Observations based on EXISTS added.")
        del data

    def observe_not_exists_attribute(self, attribute_name: str, value: str, by_time: float):
        if self.data_prepped == False:
            raise RuntimeError(f"Before the search space can be defined, one must call the 'prepare_event_log()' function.")

        if attribute_name not in self.data.columns:
            raise ValueError(f"Attribute {attribute_name} is not found in the dataset. Pick one of {self.data.columns}.")

        if by_time < 0:
            raise ValueError(f"by_time must be at least 0. You passed {by_time}.")
        
        aggregates = pd.DataFrame(columns=["case:concept:name", "observation", "time:timestamp"])

        for case in tqdm(self.data["case:concept:name"].unique(), desc = f"Observe NOT EXISTS - {attribute_name} = {value}, by_time = {by_time}"):
            subset: pd.DataFrame = copy.deepcopy(self.data[["case:concept:name", attribute_name, "time:timestamp"]][self.data["case:concept:name"] == case])
            case_start_time = subset['time:timestamp'].min()

            # Determine if the attribute value was observed before the given timing or not
            # For the subset, set starting time unit to 0
            mintime = subset['time:timestamp'].min()
            subset['time:timestamp:reduced'] = subset["time:timestamp"].apply(lambda x: x - mintime)

            # When the attribute value is not observed AND the case has taken longer than the threshold, we can add the observation at the by_time.
            if value not in subset[[attribute_name]].values and subset['time:timestamp:reduced'].max() > by_time:
                temp = pd.DataFrame({'case:concept:name' : [case], 'observation' : [f'{value} not observed within {by_time} {self.time_unit}'], 'time:timestamp' : [case_start_time + by_time]})
                aggregates = aggregates.append(temp, ignore_index=True)

            # Also add the observation if the value is observed but (the first occurence happened) after the time threshold
            if value in subset[[attribute_name]].values:
                value_observed_times = subset[subset[attribute_name] == value][["time:timestamp:reduced"]]
                value_observed_mintime = value_observed_times['time:timestamp:reduced'].min()
                if value_observed_mintime > by_time:
                    temp = pd.DataFrame({'case:concept:name' : [case], 'observation' : [f'{value} not observed within {by_time} {self.time_unit}'], 'time:timestamp' : [case_start_time + by_time]})
                    aggregates = aggregates.append(temp, ignore_index=True)

        if len(aggregates.index) > 0:
            # Add aggregates to the observations dataframe
            self.observations = self.observations.append(aggregates, ignore_index=True)

        self.arrange_observations()
        # print(f"Observations based on NOT EXISTS for attribute {attribute_name} with value {value} added.")

    def observe_and(self, attribute: str, values: set):
        if self.data_prepped == False:
            raise RuntimeError(f"Before the search space can be defined, one must call the 'prepare_event_log()' function.")

        if attribute not in self.data.columns:
            raise ValueError(f"Attribute {attribute} is not found in the dataset. Pick one of {self.data.columns}.")
            
        if not all(value in self.data[attribute].unique() for value in values):
            raise ValueError(f"Not all values of {values} were found in the data. Pick values from {self.data[attribute].unique()}")

        aggregates = pd.DataFrame(columns=["case:concept:name", "observation", "time:timestamp"])

        for case in tqdm(self.data["case:concept:name"].unique(), desc = f"Observe AND - {attribute}: {values}"):
            subset: pd.DataFrame = copy.deepcopy(self.data[["case:concept:name", attribute, "time:timestamp"]][self.data["case:concept:name"] == case])
            subset = subset[subset[attribute].isin(values)].reset_index(drop=True)
            counters = {i : 0 for i in values}

            for i in range(len(subset.index)):
                time = subset['time:timestamp'][i]
                counters[subset[attribute][i]] += 1
                counter_values = [val for key,val in counters.items()]
                if 0 not in counter_values:
                    temp = pd.DataFrame({'case:concept:name' : [case], 'observation' : [f"{values} have all occurred"], 'time:timestamp' : [time]})
                    aggregates = aggregates.append(temp)
                    for key in counters:
                        counters[key] -= 1

        if len(aggregates.index) > 0:
            # Add aggregates to the observations dataframe
            self.observations = self.observations.append(aggregates, ignore_index=True)

        self.arrange_observations()


    def observe_or(self, attribute: str, values: list):
        if self.data_prepped == False:
            raise RuntimeError(f"Before the search space can be defined, one must call the 'prepare_event_log()' function.")

        if attribute not in self.data.columns:
            raise ValueError(f"Attribute {attribute} is not found in the dataset. Pick one of {self.data.columns}.")
            
        if not all(value in self.data[attribute].unique() for value in values):
            raise ValueError(f"Not all values of {values} were found in the data. Pick values from {self.data[attribute].unique()}")
        
        aggregates = pd.DataFrame(columns=["case:concept:name", "observation", "time:timestamp"])

        for case in tqdm(self.data["case:concept:name"].unique(), desc = f"Observe OR - {attribute}: {values}"):
            subset: pd.DataFrame = copy.deepcopy(self.data[["case:concept:name", attribute, "time:timestamp"]][self.data["case:concept:name"] == case])
            subset = subset[subset[attribute].isin(values)].reset_index(drop=True)

            subset[attribute] = subset[attribute].apply(lambda x: f"One of {values} detected")
            subset = subset.rename(columns={attribute : 'observation'})

            aggregates = aggregates.append(subset)
        
        if len(aggregates.index) > 0:
            # Add aggregates to the observations dataframe
            self.observations = self.observations.append(aggregates, ignore_index=True)
        
        self.arrange_observations()

    def observe_not_exists_activity_resource_combo(self, activity: str, resource: str, by_time: float):
        if self.data_prepped == False:
            raise RuntimeError(f"Before the search space can be defined, one must call the 'prepare_event_log()' function.")
        print("observe_not_exists_activity_resource_combo -- Not supported yet. The goal would be to easily identify if an activity was executed by an unallowed resource. For example: approvals must be performed by managers.")
        pass

    def observe_directly_follows(self, activity1: str, activity2: str, negate: bool = False):
        if self.data_prepped == False:
            raise RuntimeError(f"Before the search space can be defined, one must call the 'prepare_event_log()' function.")

        # Act2 must follow Act1 without other activities begin executed. Negate turns it around: add obs when it does NOT follow immediately after.
        if activity1 not in self.data["concept:name"].unique() or activity2 not in self.data["concept:name"].unique():
            raise ValueError(f"Activity {activity1} or activity {activity2} not found in the data. Pick activities from {self.data['concept:name'].unique()}")
        
        aggregates = pd.DataFrame(columns=["case:concept:name", "observation", "time:timestamp"])

        for case in tqdm(self.data["case:concept:name"].unique(), desc = f"Observe directly follows - {activity1} => {activity2}"):
            subset: pd.DataFrame = copy.deepcopy(self.data[['case:concept:name', 'concept:name', 'time:timestamp']][self.data['case:concept:name'] == case])
            subset = subset.reset_index(drop=True)

            # For every instance of Act1, check if the next row contains Act2.
            for i in range(len(subset.index) - 1):
                if subset['concept:name'][i] == activity1:
                    if subset['concept:name'][i+1] == activity2 and not negate:
                        aggregates = aggregates.append(
                            pd.DataFrame({'case:concept:name':[case], 'observation':[activity2 + ' directly follows ' + activity1],
                            'time:timestamp':[subset['time:timestamp'][i+1]]}))
                    if negate and subset['concept:name'][i+1] != activity2:
                        aggregates = aggregates.append(
                            pd.DataFrame({'case:concept:name':[case], 'observation':[activity2 + ' did not directly follow ' + activity1],
                            'time:timestamp':[subset['time:timestamp'][i+1]]}))

        if len(aggregates.index) > 0:
            # Add aggregates to the observations dataframe
            self.observations = self.observations.append(aggregates, ignore_index=True)

        self.arrange_observations()
        # print(f"Observations based on DIRECTLY FOLLOWS for activities {activity1} followed by {activity2} added with negate = {negate}.")

    def observe_follows_within(self, activity1: str, activity2: str, margin: float, negative: bool = False):
        if self.data_prepped == False:
            raise RuntimeError(f"Before the search space can be defined, one must call the 'prepare_event_log()' function.")

        # Check if Act2 follows Act1 within a time frame, no matter if other activities were executed in between
        if activity1 not in self.data["concept:name"].unique() or activity2 not in self.data["concept:name"].unique():
            raise ValueError(f"Activity {activity1} or activity {activity2} not found in the data. Pick activities from {self.data['concept:name'].unique()}")
        
        aggregates = pd.DataFrame(columns=["case:concept:name", "observation", "time:timestamp"])

        for case in tqdm(self.data["case:concept:name"].unique(), desc = f"Observe follows within - {activity1} => {activity2}, margin = {margin}"):
            subset: pd.DataFrame = copy.deepcopy(self.data[['case:concept:name', 'concept:name', 'time:timestamp']][self.data['case:concept:name'] == case])
            subset = subset.reset_index(drop=True)

            # Filter the subset to only contain the rows for the entered activities.
            subset = subset[subset['concept:name'].isin([activity1, activity2])].reset_index(drop = True)

            # For every instance of Act1, check if the next rows contains Act2 and the timing is within the margin.
            for i in range(len(subset.index) - 1):
                if subset['concept:name'][i] == activity1:
                    ref_time = subset['time:timestamp'][i]
                    for j in range(i+1, len(subset.index)):
                        if subset['concept:name'][j] == activity2 and subset['time:timestamp'][j] <= ref_time + margin and not negative:
                            aggregates = aggregates.append(
                                pd.DataFrame({'case:concept:name':[case], 'observation':[f"{activity2} followed {activity1} within {margin} {self.time_unit}"],
                                'time:timestamp':[subset['time:timestamp'][j]]}))
                            break
                        if subset['concept:name'][j] == activity2 and subset['time:timestamp'][j] >= ref_time + margin and negative:
                            aggregates = aggregates.append(
                                pd.DataFrame({'case:concept:name':[case], 'observation':[f"{activity2} did not follow followed {activity1} within {margin} {self.time_unit}"],
                                'time:timestamp':[subset['time:timestamp'][j]]}))
                            break

        if len(aggregates.index) > 0:
            # Add aggregates to the observations dataframe
            self.observations = self.observations.append(aggregates, ignore_index=True)

        self.arrange_observations()
        # print(f"Observations based on FOLLOWS WITHIN for activities {activity1} followed by {activity2} with margin = {margin}.")

    def observe_case_delay(self, threshold: float):
        if self.data_prepped == False:
            raise RuntimeError(f"Before the search space can be defined, one must call the 'prepare_event_log()' function.")
            
        if threshold <= 0:
            raise ValueError(f"The threshold value must be higher than 0. You passed {threshold}")
            
        threshold_abs = True if threshold > 1 else False

        # Determine the durations of the different traces
        aggregates = self.data[['case:concept:name', 'time:timestamp']].groupby('case:concept:name').agg({'time:timestamp': ['min', 'max']})
        aggregates.columns = aggregates.columns.to_flat_index()
        aggregates = aggregates.assign(diff = lambda x: x[('time:timestamp', 'max')] - x[('time:timestamp', 'min')])

        if threshold_abs:
            print(f"Absolute threshold set to {threshold}")
        else:
            print(f"Relative threshold set to {threshold * 100}% of max case duration")
            max_duration = max(aggregates['diff'])
            threshold = max_duration * threshold
        
        aggregates = aggregates.assign(
            delayed = lambda x: x['diff'] >= threshold,
            time = lambda x: x[('time:timestamp', 'min')] + threshold
        )
        aggregates = aggregates.reset_index()
        aggregates = aggregates.rename(columns={'index' : 'case:concept:name', 'delayed' : 'observation'})
        aggregates = aggregates[['case:concept:name', 'observation', 'time']][aggregates['observation'] == True]
        aggregates = aggregates.assign(observation = "Case Delayed")
        aggregates = aggregates.rename(columns={'time' : 'time:timestamp'})

        # Add artificial events to the data
        self.observations = self.observations.append(aggregates, ignore_index=True)
        self.arrange_observations()

        print("Effect by delay artificially added.")

    def observe_exists_single_value(self, attribute: str, value: str):
        if self.data_prepped == False:
            raise RuntimeError(f"Before the search space can be defined, one must call the 'prepare_event_log()' function.")

        if attribute not in self.data.columns:
            raise ValueError(f"Attribute {attribute} is not found in the dataset. Pick one of {self.data.columns}.")
            
        if not value in self.data[attribute].unique():
            raise ValueError(f"Value {value} was not found in the column {attribute}. Pick values from {self.data[attribute].unique()}")

        subset: pd.DataFrame = copy.deepcopy(self.data[['case:concept:name', 'concept:name', 'time:timestamp']][self.data[attribute] == value])
        subset = subset.rename(columns={attribute : 'observation'})
        subset = subset.reset_index(drop=True)

        # Add events to the data
        self.observations = self.observations.append(subset, ignore_index=True)
        self.arrange_observations()

        print(f"Single {attribute} value observations {value} added.")

    def filter_search_space(self, threshold: float):
        # What to filter out?
        ### - Observations which can be mutually exclusive, like idling times between two specific activities (not supported yet)
        ### - Observations which are only made in less than [threshold]% of the cases
        
        if threshold != None and (threshold > 1 or threshold < 0):
            raise ValueError(f"The threshold value must take a value between 0 and 1. {threshold} was passed.")

        if 'observation' not in self.observations.columns:
            raise RuntimeError(f"The set of hypotheses can only be filtered after it is created. Run 'define_search()' first.")
        
        # Get the different observations and count them to relative frequencies, filter on the threshold
        self.observations = self.observations[self.observations['observation'].map(self.observations['observation'].value_counts(normalize = True)) > threshold]

        print(f"Observations filtered based on minimum frequency of {threshold * 100}%")

    def arrange_observations(self):
        self.observations = self.observations.sort_values('time:timestamp', ascending=True).reset_index(drop=True)

    def filter_observations_NaN(self):
        self.observations = self.observations[self.observations['observation'].notnull()]

    def export_data_to_csv(self, path: str):
        self.data.to_csv(path, index=False)

    def export_observations_to_csv(self, path: str):
        self.arrange_observations()
        self.observations.to_csv(path, index=False)

    def __str__(self) -> str:
        return self.observations.__str__()