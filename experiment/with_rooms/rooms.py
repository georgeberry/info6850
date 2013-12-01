from gevent import monkey
monkey.patch_all()
from flask import Flask, render_template, Response, request
import werkzeug.serving
from socketio import socketio_manage
from socketio.namespace import BaseNamespace
from socketio.mixins import RoomsMixin
from socketio.server import SocketIOServer

# app #
app = Flask(__name__)


# views #
@app.route('/')
def hello():
    return render_template('intro.html')


@app.route('/<path:room>')
def room_picker(room):
    context = {'room': room} #pass args to the template renderer for the specific room
    return render_template('room.html', **context)


#  #


# namespace #
class ChatNamespace(BaseNamespace, RoomsMixin):
    '''
    Rooms. Sorts people automatically into a random room.
    You have to enter a name.
    Broadcast only to your room.
    Logs messages in a dictinoary keyed by rooms.
    '''

    sockets = {} #tracks sockets in the namespace
    rooms = range(10)
    room_dict = {} #key: users, value: room

    def recv_connect(self):
        print 'connected'
        self.sockets[id(self)] = self

    def recv_disconnect(self):
        super(ChatNamespace, self).disconnect()
        print 'disconnected'

    def on_name(self, name):
        self.session['name'] = name #session is dictionary associated with socket
        self.room_dict[name] = 1 #put individual in a room
        return redirect(url_for('room_picker', room = self.room_dict[name])) #redirect; may have to do this a different way

    def on_user_message(self, msg):
        self.emit_to_room(self.room, 'msg_to_room', self.session['name'], msg)

    












# socket io server always defaults here #
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