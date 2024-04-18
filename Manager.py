import asyncio
import time
import aio_pika
import optuna
from optuna import Trial, Study
from optuna.distributions import BaseDistribution
from typing import Callable, Dict, List
from dc1.distributed.Message import Task, ParamMessage
from copy import copy

import nest_asyncio
nest_asyncio.apply()

class OptunaDistributedManager:
    def __init__(self, rmq_serv_ip, study: Study, param_distribution_callable: Callable[[Trial], Dict], init_params: ParamMessage):
        self.init_params = init_params
        self._study = study
        self.param_dist_fn = param_distribution_callable
        self.rmq_serv_ip = rmq_serv_ip
        self.completed_trials = 0
        self.active_trials = dict()

    def ask(self):
        trial = self._study.ask()
        return trial, self.param_dist_fn(trial)

    def tell(self, trial, eval=None, state=None):
        trial = self._study.tell(trial, values=eval, state=state)
        (task, executor) = self.active_trials[trial.number]
        if state == optuna.trial.TrialState.COMPLETE or optuna.trial.TrialState.PRUNED:
            print(f"ending trial {trial.number}")
            self.completed_trials += 1
            executor.cancel()
            del self.active_trials[trial.number]
        if state == optuna.trial.TrialState.FAIL:
            print(f"cancelling trial {trial.number}")
            executor.cancel()
            del self.active_trials[trial.number]

    def optimize(self, n_trials, max_conc_trials=5):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.run_optimization(n_trials, max_conc_trials))

    async def run_optimization(self, n_trials, max_conc_trials=5):
        connection = await aio_pika.connect_robust(self.rmq_serv_ip)
        async with connection:
            channel = await connection.channel()
            task_queue = await channel.declare_queue("task_queue")
            await task_queue.purge()
            task_display_timer = 30
            init_ = time.time()
            while self.completed_trials < n_trials:
                try:
                    if len(self.active_trials) < max_conc_trials and (self.completed_trials + len(self.active_trials) < n_trials):
                        trial, params = self.ask()
                        param_message = copy(self.init_params)
                        param_message.params = params
                        task = Task(trial, channel, param_message, "task_queue", self.tell, patience_time_out=60*20)
                        await task.run()
                        t = asyncio.create_task(coro = task.handle_response_if_any())
                        self.active_trials[trial.number] = (task, t)
                        print("Task added!")
                    await asyncio.sleep(0)
                    if(time.time() - init_ > task_display_timer):
                        init_ = time.time()
                        for trial, (t, task) in self.active_trials.items():
                            print(trial, "Task Complete: ",task.done(), "Task Cancelled: ",task.cancelled())
                except asyncio.CancelledError as e:
                    print("Some task got cancelled", e.with_traceback)
            return self._study.best_params, self._study.best_trial

if __name__ == "__main__":
    st = optuna.create_study()
    def param_distr(trial):
        params  = {"x": trial.suggest_float("x", 0, 1)}
        return params
    dist_study = OptunaDistributedManager("tcp://localhost:5672", st, param_distr, ParamMessage(None, [0, 1], [2, 3], "hi"))
    result = dist_study.optimize(100)
    print(result)

class StudyManager:
    def __init__(self, study: Study, param_dist_fn, init_param_message, n_trials, max_conc):
        self._study = study
        self.active_trials = dict()
        self.param_dist_fn = param_dist_fn
        self.completed_trials = 0
        self.max_trials = n_trials
        self.max_conc = max_conc
        self.init_param = init_param_message

    def ask(self):
        trial = self._study.ask()
        return trial, self.param_dist_fn(trial)

    def tell(self, trial, eval=None, state=None):
        trial = self._study.tell(trial, values=eval, state=state)
        (task, executor) = self.active_trials[trial.number]
        if state == optuna.trial.TrialState.COMPLETE or optuna.trial.TrialState.PRUNED:
            print(f"ending trial {trial.number}")
            self.completed_trials += 1
            executor.cancel()
            self.active_trials.pop(trial.number, None)
        if state == optuna.trial.TrialState.FAIL:
            print(f"cancelling trial {trial.number}")
            executor.cancel()
            self.active_trials.pop(trial.number, None)

    def is_complete(self):
        return self.completed_trials >= self.max_trials

    def can_add(self):
        return self.max_conc > len(self.active_trials) and self.max_trials >= self.completed_trials + len(self.active_trials)

    def create_task(self, channel):
        trial, params = self.ask()
        param_message = copy(self.init_param)
        param_message.params = params
        return Task(trial, channel, param_message, "task_queue", self.tell, patience_time_out=60*20)

    def register_task(self, taskmanager, task):
        self.active_trials[taskmanager.trial.number] = (taskmanager, task)

    def best_pt(self):
        return self._study.best_params, self._study.best_trials

    def best_param(self):
        return self._study.best_params

    def purge_dead_trials(self):
        to_purge = []
        for t_n, (trial, task) in self.active_trials.items():
            if task.done():
                to_purge.append(t_n)
        for n in to_purge:
            print(f"purging {n}")
            self.active_trials[n][1].cancel()
            self.active_trials.pop(n, None)

    def print_statistics(self):
        print(f"Studies done: {self.completed_trials} active: {len(self.active_trials)} total: {self.max_trials}")



class MultiStudyManager:
    def __init__(self, rmq_addr):
        self.rmq_addr = rmq_addr
        self.study_managers: List[StudyManager]= []

    def add_study(self, study, param_dist_callable, init_params, n_trials = 100, max_conc = 5):
        self.study_managers.append(StudyManager(study, param_dist_callable, init_params, n_trials, max_conc))

    def optimize(self):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.run_all_studies())

    async def run_all_studies(self):
        connection = await aio_pika.connect_robust(self.rmq_addr)
        async with connection:
            channel = await connection.channel()
            task_queue = await channel.declare_queue("task_queue")
            await task_queue.purge()

            task_display_timer = 300  # logging stuff
            init_ = time.time()

            active = self.study_managers
            while active:
                try:
                    active = [*filter(lambda x: not x.is_complete(), self.study_managers)]
                    for study in active:
                        if study.can_add():
                            task = study.create_task(channel)
                            await task.run()
                            t = asyncio.create_task(coro = task.handle_response_if_any())
                            study.register_task(task, t)
                            print("Task added!")
                        await asyncio.sleep(0)
                        # Debug
                    if(time.time() - init_ > task_display_timer):
                        init_ = time.time()
                        for study in active:
                            study.print_statistics()
                            study.purge_dead_trials()
                except asyncio.CancelledError as e:
                    print("Some task got cancelled", e.with_traceback)
                except:
                    print("unknown error, continuing")
            results = [*map(lambda x : (x.best_param(), x.init_param), self.study_managers)]
            return results