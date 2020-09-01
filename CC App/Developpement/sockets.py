import socketio


#class ClientSocket(socketio.Client):

class ClientSocket():

    def __init__(self, **kwargs):
        super(ClientSocket,self).__init__(**kwargs)
        self.init_handlers()

    def init_handlers(self):
        @self.event
        def connect():
            print("[INFO] Connected to server successfully.")

        @self.event
        def connect_error():
            print("[Connection Error] Couldn't connect to Server")

        @self.event
        def disconnect():
            print("[INFO] Socket disconnected.")

    def add_handlers(self,handlers):
        for event,event_handler in handlers:
            self.on(event,handler=event_handler)   
            

        # @self.on('server-error')
        # def disconnect():
        #     print("[INFO] Socket disconnected.")    
        # @self.on('process-done')
        # def disconnect():
        #     print("[INFO] Socket disconnected.")    
        # @self.on('send-image')
        # def disconnect():
        #     print("[INFO] Socket disconnected.")    


#
#npx json-server --watch db.json            