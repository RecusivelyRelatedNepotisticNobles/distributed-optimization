from collections import namedtuple
from uuid import uuid4
import traceback
import pickle
import time
import pika
import pika.adapters.blocking_connection
from Message import ParamSuggestion, PruneResponse, CancelOpt, AcceptedWorker, DeclinedWorker
from Message import TaskResponseAccepted, TaskResponseIntermediate, TaskResponseCancel
from Message import TaskResponsePruneQuery, TaskResponseCompleted
from Message import ParamMessage

class Communicator:
    def __init__(self, in_, out_, channel: pika.BlockingConnection):
        self._in = in_
        print(f"listening to {self._in}")
        self._out = out_
        print(f"sending to {self._out}")
        self.channel = channel

    def get_message(self):
        header, properties, body = self.channel.basic_get(queue=self._in, auto_ack=True)
        if header:
            body = pickle.loads(body)
        return header, body, properties

    def send_message(self, body, target):
        body = pickle.dumps(body)
        self.channel.basic_publish(
            exchange='',
            routing_key=target,
            body=body
        )

    def send_to_out(self, body):
        body = pickle.dumps(body)
        self.channel.basic_publish(
            exchange='',
            routing_key=self._out,
            body=body
        )

    def clean_up(self):
        # self.channel.queue_delete(queue=self._out)
        pass

ReportingTrial = namedtuple("ReportingTrial", ["should_prune", "report"])

class TaskHandler:
    def __init__(self, communicator, termination_callback, param_message: ParamMessage):
        self.communicator = communicator
        self.termination_callback = termination_callback
        self.params = param_message.params
        self.train_idxs = param_message.idx_train
        self.test_idxs = param_message.idx_eval
        self.model_name = param_message.model_type
        self.iscomplete = False


    def handle_task(self, model_from_params, model_executor, label_creator):
        model = model_from_params(self.model_name, self.params, label_creator(self.train_idxs))
        try:
            reporting_trial = ReportingTrial(self.should_prune, self.intermediate)
            final_eval = model_executor(model, self.params, reporting_trial, self.train_idxs, self.test_idxs)
            self.complete_trial(final_eval)
        except TrialPruned:
            print("Trial Pruned")
            return
        except Exception as e:
            print(f"trial ended because {traceback.format_exc()}")
            self.fail_trial("execution stopped unexpectedly")
        finally:
            del model
        
    def fail_trial(self, reason="unknown"):
        self.communicator.send_to_out(TaskResponseCancel(reason))

    def validate_expected_type_or_quit(self, message_body, expected_type):
        if(type(message_body) is not expected_type):
            self.termination_callback()

    def wait_for_should_prune(self, timeout):
        starting_time = time.time()
        while(starting_time - time.time() < timeout):
            header, body, props = self.communicator.get_message()
            if header:
                self.validate_expected_type_or_quit(body, PruneResponse)
                return body.should_prune
            time.sleep(0.5)
        return None

    def check_if_quit_message(self):
        header, body, props = self.communicator.get_message()
        if header:
            if type(body) is CancelOpt:
                self.termination_callback()

    def complete_trial(self, final_eval):
        self.communicator.send_to_out(TaskResponseCompleted(final_eval))
        self.iscomplete = True

    def intermediate(self, eval, step):
        self.communicator.send_to_out(TaskResponseIntermediate(eval, step))
        self.check_if_quit_message()

    def should_prune(self):
        self.communicator.send_to_out(TaskResponsePruneQuery())
        should_prune = self.wait_for_should_prune(100)
        if should_prune is None or should_prune:
            self.iscomplete = True
            raise TrialPruned
        return should_prune
    
    def clean_up(self, *args):
        if not self.iscomplete:
            self.fail_trial()
        self.communicator.clean_up()

class TrialPruned(Exception):
    pass

class TaskQueueNegotiator:
    def __init__(self, tq_name, channel: pika.adapters.blocking_connection.BlockingChannel):
        self.communicator = Communicator(tq_name, None, channel)
        self.active_task = None
        self.channel = channel

    def accept_and_wait(self, time_out, clean_up):
        header, body, props = self.communicator.get_message()
        if header:
            print("negotiating task")
            param_message: ParamSuggestion= body
            new_in_channel = str(uuid4())
            self.channel.queue_declare(queue=new_in_channel, auto_delete=True)
            new_comm = Communicator(new_in_channel, props.reply_to, self.channel)
            new_comm.send_to_out(TaskResponseAccepted(new_in_channel))
            time_in = time.time()
            while (time.time() - time_in < time_out):
                header, body, params = new_comm.get_message()
                if header:
                    if type(body) is AcceptedWorker:
                        print(f"Task has been Accepted!, {param_message.pm.params}")
                        return TaskHandler(new_comm, clean_up, param_message.pm)
                    if type(body) is DeclinedWorker:
                        return
                time.sleep(0.5)
        return


class Worker:
    def __init__(self, amqp_url, model_from_params, model_executor, label_creator):
        url_params = pika.URLParameters(amqp_url)
        url_params.heartbeat = 60*30 # Heartbeat set at 30 minutes, if non-responsive for 30 minutes it will get closed.
        self.connection = pika.BlockingConnection(url_params)
        self.channel = self.connection.channel()
        tq_name = 'task_queue'
        self.tq = self.channel.queue_declare(queue=tq_name)
        self.tq_neg = TaskQueueNegotiator(tq_name, self.channel)
        self.active_task = None
        self.model_from_params = model_from_params
        self.model_executor = model_executor
        self.label_creator = label_creator

    def run(self):
        try:
            while(True):
                self.active_task = self.tq_neg.accept_and_wait(100, self.clean_up)
                if self.active_task:
                    self.active_task.handle_task(self.model_from_params, self.model_executor, self.label_creator)
                    self.active_task.clean_up()
                    self.active_task = None
                time.sleep(0.1)
        except Exception as e:
            print(f"exception, {traceback.format_exc()}")
        finally:
            self.clean_up()

    def clean_up(self):
        if self.active_task:
            self.active_task.clean_up()
            self.active_task = None
