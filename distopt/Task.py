from enum import Enum
import traceback
from uuid import uuid4
import aio_pika
from aio_pika import RobustChannel
import asyncio
import pickle
import time

import aiormq
from optuna.trial import TrialState, Trial

import Message as m

class TaskState(Enum):
    Starting = 1
    Started = 2
    Completed = 3

class Task:
    def __init__(self, trial, channel: RobustChannel, pm: m.ParamMessage, task_queue_name, tell_callback, patience_time_out):
        self.uuid = str(uuid4())
        self.trial = trial
        self.channel = channel
        self.task_state = TaskState.Starting
        self.tq_name = task_queue_name
        self.pm = pm
        self.out_name = None
        self.task_queue = None
        self.response_queue = None
        self.patience_timeout = patience_time_out
        self.last_ping = None
        self.tell_cb = tell_callback
        return None

    async def run(self):
        self.response_queue = await self.channel.declare_queue(self.uuid)
        print(f"Listening on {self.uuid}")
        self.task_queue = await self.channel.declare_queue(self.tq_name)
        await self.post_task()
        await self.response_queue.purge()

    async def handle_response_if_any(self):
        try:
            while not self.is_completed():
                if self.task_state == TaskState.Started and time.time() - self.last_ping > self.patience_timeout:
                    self.report_failure()
                if self.response_queue:
                    try:
                        message = await self.response_queue.get(timeout=1)
                        print("handling response")
                        body = pickle.loads(message.body)
                        await self.handle_message(body)
                        await message.ack()
                    except aio_pika.exceptions.QueueEmpty:
                        await asyncio.sleep(0)
                        continue
                    except asyncio.exceptions.TimeoutError:
                        #sleep for 3 seconds and retry
                        await asyncio.sleep(3)
                        continue
                    except aio_pika.exceptions.ChannelPreconditionFailed:
                        await asyncio.sleep(3)
                        continue
                    except aiormq.exceptions.ChannelInvalidStateError:
                        await asyncio.sleep(3)
                        continue
                await asyncio.sleep(0)
            return
        except Exception as e:
            print(traceback.format_exc())
            return

    def report_failure(self):
        print("reporting failure")
        self.tell_cb(self.trial, state=TrialState.FAIL)
        self.task_state = TaskState.Completed

    async def post_task(self):
        message_body = pickle.dumps(m.ParamSuggestion(self.uuid, self.pm))
        message = aio_pika.Message(body=message_body)
        message.correlation_id = self.uuid
        message.reply_to = self.uuid
        await self.channel.default_exchange.publish(message, routing_key=self.tq_name)

    def create_message(self, body):
        body = pickle.dumps(body)
        message = aio_pika.Message(body=body)
        message.correlation_id = str(uuid4())
        message.reply_to = self.uuid
        return message

    async def reply(self, message):
        if self.out_name:
            message.correlation_id = str(uuid4())
            message.reply_to = self.uuid
            await self.channel.default_exchange.publish(message, routing_key=self.out_name)
        else:
            raise Exception("Trying to reply when no connection to worker has been established.")
    
    async def send_to_target(self, message, target):
        message.correlation_id = str(uuid4())
        message.reply_to = self.uuid
        await self.channel.default_exchange.publish(message, routing_key=target)

    async def handle_message(self, body):
        print(f"handling message, {body}")
        if self.task_state == TaskState.Started:
            self.last_ping = time.time()
        match(body.response_type):
            case(m.TaskResponse.Accepted):
                print(f"trial : {self.trial.number} Has been accepted")
                if not self.out_name:
                    self.last_ping = time.time()
                    self.out_name = body.uuid
                    message = self.create_message(m.AcceptedWorker(self.uuid))
                    await self.reply(message)
                    self.task_state = TaskState.Started
                else:
                    message = self.create_message(m.DeclinedWorker(self.uuid))
                    self.send_to_target(message, body.uuid)
            case(m.TaskResponse.IntermediateValue):
                self.trial.report(body.intermediate, step=body.step)
            case(m.TaskResponse.PruneQuery):
                sp = False
                response = m.PruneResponse(self.uuid, False)
                if self.trial.should_prune():
                    sp = True
                    response = m.PruneResponse(self.uuid, True)
                    print(f"{self.trial.number} pruned")
                message = self.create_message(response)
                await self.reply(message)
                if sp:
                    self.tell_cb(self.trial, state=TrialState.PRUNED)
            case(m.TaskResponse.Completed):
                print(f"trial: {self.trial.number} Has ended with eval: {body.final_eval}")
                self.task_state = TaskState.Completed
                self.tell_cb(self.trial, body.final_eval, state=TrialState.COMPLETE)
            case(m.TaskResponse.ExecutionCancelled):
                print(f"trial: {self.trial.number} was cancelled, reason: {body.reason}")
                self.report_failure()

    def is_completed(self):
        return self.task_state == TaskState.Completed

    def clean(self):
        if self.out_name:
            self.response_queue = None