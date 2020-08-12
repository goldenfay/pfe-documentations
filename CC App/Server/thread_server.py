from flask import Flask
import threading
from werkzeug.serving import make_server
import ctypes
import signal
class ServerThread(threading.Thread):

    def __init__(self, srv: Flask,port=4000):
        threading.Thread.__init__(self)
        self.srv = make_server('127.0.0.1', 4000, srv)
        # self.srv = srv
        self._stopper = threading.Event()
        self.ctx = srv.app_context()
        # self.ctx.push()

    def run(self):
        print('starting process server')

        try:
            # self.srv.run('127.0.0.1', 4000)
            self.srv.serve_forever()
        except:
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

