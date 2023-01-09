# AITIA-PM
Source code for the AITIA-PM algorithm. See paper entitled "Root Cause Analysis in Process Mining with Probabilistic Temporal Logic"

### Contents of the repository
* `\Data` - Contains the modified csv file to contain case delays as observations.
* `\R` - Contains the source R file to compute the false discovery rates.
* `\Output` - Contains the output files from AITIA-PM.
* `main.py` - The python source code to apply AITIA-PM on a dataset.
* `Hypothesizer.py` - The python class built to define the search space.
* `Inference.py` - The python class to identify cause-effect relations.