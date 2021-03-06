from gevent import monkey
monkey.patch_all()
from flask import Flask, render_template, Response, request
import werkzeug.serving
from socketio import socketio_manage
from socketio.namespace import BaseNamespace
from socketio.server import SocketIOServer

# app #
app = Flask(__name__)

# views #

@app.route('/')
def hello():
    return render_template('main.html')


# namespace #
class ChatNamespace(BaseNamespace, RoomsMixin):
    '''
    No rooms yet. 
    Just an entry box and a box that dynamically displays messages and timestamps.
    Anyone connecting should see all subsequent messages
    '''

    sockets = {}
    messages = []

    def recv_connect(self):
        print 'connected'
        self.emit("previous", self.messages)
        self.sockets[id(self)] = self

    def recv_disconnect(self):
        super(ChatNamespace, self).disconnect()
        print 'disconnected'

    def on_text(self, text):
        self.messages.append(text)

        for ws in self.sockets.values():
            ws.emit("text back", text)



#socket io server always defaults here
@app.route('/socket.io/<path:rest>')
def push_stream(rest):
    try:
        socketio_manage(request.environ, {'/chat': ChatNamespace}, request)
    except:
        app.logger.error('Exception while handling socketio connection', exc_info=True)
    return 'chat'



# serve #
@werkzeug.serving.run_with_reloader
def run_dev_server():
    app.debug = True
    port = 6020
    SocketIOServer(('', port), app, resource='socket.io').serve_forever()


# run program #
if __name__ == '__main__':
    run_dev_server()