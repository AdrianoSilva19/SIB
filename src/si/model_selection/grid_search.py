
from si.data.dataset import Dataset
from typing import Callable,Dict,List
from si.model_selection.cross_validate import cross_validate
import itertools


def grid_search_cv(model,dataset:Dataset,parameter_grid:Dict[str, List[float]],scoring:Callable=None,cv:int=3,test_size:float=0.2):
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    scores = []
    # for each combination
    combinations = itertools.product(*parameter_grid.values())

    for combination in combinations:
        # parameters
        parameters = {}

        # set parameters
        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # add parameter configuration
        score["parameters"] = parameters

        # add score to scores
        scores.append(score)

    return scores