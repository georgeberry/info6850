from gevent import monkey
monkey.patch_all()
from flask import Flask, render_template, Response, request, redirect, url_for
import werkzeug.serving
from socketio import socketio_manage
from socketio.namespace import BaseNamespace
from socketio.mixins import RoomsMixin
from socketio.server import SocketIOServer
from flask.ext.sqlalchemy import SQLAlchemy
import time


# app #

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/chat.db'
db = SQLAlchemy(app)
usrs = {} #user information keyed by hash



# views #

@app.route('/')
def hello():
    return render_template('intro.html')


@app.route('/validate', methods=['POST'])
def validate_or_kick():
    '''user id input here; check for validity
        redirect to an appropriate chatroom

        can be setup to take inputs from a bunch of form-based pages and route accordingly
    '''

    #for name form
    usr_id = request.form['name']
    usrs.append(usr_id)
    if lookup(usr_id):
        print 'looked up'
        return redirect(url_for('room_picker', room = 'one'))
    else:
        return redirect(url_for('hello'))

    #for instructions & instruction q/a

    #for q1 form

    #for chat

    #for q2 form


# build the rooms from unique identifiers #
@app.route('/<path:room>')
def room_picker(room):
    context = {'room': room} #pass args to the template renderer for the specific room
    return render_template('room.html', **context)


# one page form w/ questions, answers written to database #
@app.route('/q1/<path:usr_id>')
def first_questions(usr_id):
    return render_template('questions1.html')


# one page form w/ questions, answers written to database #
@app.route('/q2/<path:usr_id>')
def second_questions(usr_id):
    return render_template('questions2.html')


# instructions page, with a short written answer section, saved to database #
@app.route('/i/<path:usr_id>')
def instructions(usr_id):
    return render_template('instructions.html')


# db models #


# helpers #

#need to keep chat namespace dicts in order
#easiest way to key is by id of namespace instance
#
def lookup(name):
    #if name in name dict
    return True


#join or create a room
def join_or_create():
    pass
    



# namespace #

class ChatNamespace(BaseNamespace, RoomsMixin):
    '''
    Rooms. Sorts people automatically into a random room.
    You have to enter a name.
    Broadcast only to your room.
    Logs messages in a dictionary keyed by rooms.
    '''
    session_by_id = {}
    namespace_local_info = {}
    room_dict = {} #key: room, value: users
    sockets = {} #tracks sockets in the namespace

    def __init__(self, *args, **kwargs):
        request = kwargs.get('request', None)
        self.ctx = None #pulls the context out manually
        if request:
            self.ctx = app.request_context(request.environ)
            self.ctx.push()
            app.preprocess_request()
            del kwargs['request']
        super(ChatNamespace, self).__init__(*args, **kwargs)
        if not hasattr(self.socket, 'rooms'):
            self.socket.rooms = set() # a set of simple strings

    def recv_connect(self):
        print 'connect', self
        self.sockets[id(self)] = self #for easy sending
        self.session_by_id[id(self)] = {'user': None, 'room': None}
        self.emit('connect')

    def recv_disconnect(self, *args, **kwargs):
        print 'begin disconnect'
        if self.ctx:
            try:
                self.ctx.pop()
            except:
                'normal teardown nonetype'
        self.disconnect(silent=True)

    def on_username(self, usr):
        room = 'one' #temp
        print 'username', self
        self.session['name'] = usr
        self.session_by_id[id(self)]['user'] = usr
        self.session_by_id[id(self)]['room'] = str(room)
        self.room_dict[room] = usr #put individual in a room
        self.emit("room_url", url_for('room_picker', room = room))

    def on_join(self, room):
        print 'joined'
        print usrs, self.ns_name, room
        self.join(str(room))

    def on_user_message(self, msg):
        #assumes a max of 1 room
        print 'message away!'
        self.emit_to_room('one', 'msg_to_room', usrs[0], msg)


# socket io server always defaults here #

@app.route('/socket.io/<path:rest>')
def push_stream(rest):
    try:
        real_request = request._get_current_object()
        socketio_manage(request.environ, {'/chat': ChatNamespace}, request=real_request)
    except:
        app.logger.error('Exception while handling socketio connection', exc_info=True)
    return Response()


# serve #

@werkzeug.serving.run_with_reloader
def run_dev_server():
    app.debug = True
    port = 6020
    SocketIOServer(('', port), app, resource='socket.io').serve_forever()


# run program #

if __name__ == '__main__':
    run_dev_server()