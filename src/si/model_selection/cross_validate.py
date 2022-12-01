
from si.data.dataset import Dataset
from typing import Callable
from si.model_selection.split import train_test_split
import random

def cross_validate(model,dataset:Dataset,scoring:Callable=None,cv:int=3,test_size:float=0.2):
    scores_dict = {"seeds": None, "train": None, "test": None}

    for i in range(cv):
        # get random state seed
        seed = random.randint(0, 1000)
        scores_dict["seed"].append(seed)

        # split dataset
        train, test = train_test_split(dataset=dataset, test_size=test_size, random_state=seed)

        # fit the model to the train set
        model.fit(train)

        #score the model on the test dataset
        if scoring is None:

            # store the train score
            scores_dict["train"].append(model.score(train))

            # store the test score
            scores_dict["test"].append(model.score(test))

        else:
            y_train = train.y
            y_test = test.y

            # store the train score
            scores_dict["train"].append(scoring(y_train, model.predict(train)))

            # store the test score
            scores_dict["test"].append(scoring(y_test, model.predict(test)))



    return scores_dict