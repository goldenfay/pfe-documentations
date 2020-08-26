from flask import Flask
import threading
from werkzeug.serving import make_server
import ctypes
import signal
import dill

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class ServerThread(threading.Thread):

    def __init__(self, srv: Flask,port=4000):
        threading.Thread.__init__(self)
        self.srv = make_server('127.0.0.1', port, srv)
        # self.srv = srv
        self._stopper = threading.Event()
        self.ctx = srv.app_context()
        # self.ctx.push()

    def run(self):
        print('[ServerThread] Starting process server')

        
        # self.srv.run('127.0.0.1', 4000)
        self.srv.serve_forever()
    
        self._stopper.set()

        # self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()
        # print('shutted down')
        self._stopper.set()
        # self._stop()
        print('[Server] Terminated')

    def stopped(self):
        return self._stopper.is_set()

    def get_id(self):

            # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                         ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')
        raise Exception('Forcing thread to terminate')    


def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))
