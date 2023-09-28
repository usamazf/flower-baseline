"""A module to handle experiment logging and tracking."""
import os
import numpy as np

class ExperimentManager():
    """Logic to store hyperparameters and results of an experiment."""
    
    def __init__(
            self, 
            *, 
            experiment_id: str, 
            hyperparameters: dict = None,
        ):
        self.experiment_id = experiment_id
        self.hyperparameters = hyperparameters
        self.results = dict()
        self.parameters = dict()

    def __str__(self):
        strrep = "Hyperparameters: \n"
        for key, value in self.hyperparameters.items():
            strrep += " - " + key + " "*(24-len(key)) + str(value) + "\n"
        return strrep

    def __repr__(self):
        return self.__str__()
    
    def log(self, update_dict, printout=False, override=False):
        
        # update a results dictionary
        for key, value in update_dict.items(): 
            if (not key in self.results) or override:
                self.results[key] = [value]
            else:
                self.results[key] += [value]

        if printout:
            print(update_dict)
    
    def save_parameters(self, parameters):
        self.parameters = parameters
    
    def to_dict(self):
        """A module to convert the current experiment to a dictionary."""
        return {
            "hyperparameters" : self.hyperparameters, 
            "parameters" : self.parameters, 
            "results": self.results,
        }

    def from_dict(self, input_dict):
        self.hyperparameters = input_dict["hyperparameters"][np.newaxis][0]
        self.parameters = input_dict["parameters"][np.newaxis][0]
        self.results = input_dict["results"][np.newaxis][0]
    
    def save_to_disc(self, path, filename, verbose=True):
        results_numpy = {key : np.array(value)for key, value in self.to_dict().items()}

        if not os.path.exists(path):
            os.makedirs(path)
        
        np.savez(path+filename, **results_numpy) 
        if verbose:
            print("Saved results to ", path+filename+".npz")        

    def load_from_disc(self, path, filename, verbose=True):
        results_dict = np.load(path+filename, allow_pickle=True)
        self.from_dict(results_dict)

        if verbose:
            print("Loaded results from "+path+filename)
