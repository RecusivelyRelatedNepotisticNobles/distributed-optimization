from collections import namedtuple
from enum import Enum

ParamSuggestion = namedtuple("ParamSuggestion", ["uuid", "pm"])
PruneResponse = namedtuple("PruneResponse", ["uuid", "should_prune"])
CancelOpt = namedtuple("CancelOpt", ["uuid"])
AcceptedWorker = namedtuple("AcceptedWorker", ["uuid"])
DeclinedWorker = namedtuple("DeclinedWorker", ["uuid"])

class TaskResponse(Enum):
    Accepted = 1
    IntermediateValue = 2
    PruneQuery = 3
    Completed = 4
    ExecutionCancelled = 5

class TaskResponseBase:
    def __init__(self, response_type:TaskResponse):
        self.response_type = response_type

class TaskResponseAccepted(TaskResponseBase):
    def __init__(self, uuid):
        super().__init__(TaskResponse.Accepted)
        self.uuid = uuid

class TaskResponseIntermediate(TaskResponseBase):
    def __init__(self, intermediate_value, step):
        super().__init__(TaskResponse.IntermediateValue)
        self.intermediate = intermediate_value
        self.step = step

class TaskResponsePruneQuery(TaskResponseBase):
    def __init__(self):
        super().__init__(TaskResponse.PruneQuery)

class TaskResponseCancel(TaskResponseBase):
    def __init__(self, reason):
        super().__init__(TaskResponse.ExecutionCancelled)
        self.reason = reason

class TaskResponseCompleted(TaskResponseBase):
    def __init__(self, final_eval):
        super().__init__(TaskResponse.Completed)
        self.final_eval = final_eval

class ParamMessage:
    def __init__(self, params, idx_train, idx_eval, model_type):
        self.params = params
        self.idx_train = idx_train
        self.idx_eval = idx_eval
        self.model_type = model_type
