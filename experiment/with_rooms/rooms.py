'''
@by George Berry (geb97@cornell.edu)
@Cornell Dpt of Sociology (Social Dynamics Lab)
@December 2013
'''

from gevent import monkey
monkey.patch_all()
from flask import Flask, render_template, Response, request, redirect, url_for
import werkzeug.serving
from socketio import socketio_manage
from socketio.namespace import BaseNamespace
from socketio.mixins import RoomsMixin
from socketio.server import SocketIOServer
from flask.ext.sqlalchemy import SQLAlchemy
import cPickle as pickle
from hashlib import md5
import sys
from init_db import gen_names

import time

# app #

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/chat.db'
db = SQLAlchemy(app)


# load names or get names #
try:
    users, rooms = pickle.load(open('db/names.p', 'rb'))
except:
    users.rooms = gen_names(500, 6, 8)


# views #

@app.route('/') #1
def hello():
    return render_template('intro.html')


@app.route('/validate', methods=['POST']) #glues pages together
def validate_or_kick():
    '''user id input here; check for validity
        redirect to an appropriate chatroom

        can be setup to take inputs from a bunch of form-based pages and route accordingly
    '''
    print request.form

    if 'name' in request.form.keys():
        usr_id = str(request.form['name'])
        try: 
            if 'logons' in users[usr_id].keys():
                users[usr_id]['logons'] += 1
            elif 'logons' not in users[usr_id].keys():
                users[usr_id]['logons'] = 1
        except:
            print 'bad login attempt with code', usr_id
            return redirect(url_for('hello'))
        users[usr_id]['md5'] = md5(usr_id).hexdigest()
        return redirect(url_for('instructions', usr_id = users[usr_id]['md5']))

    if 'instructions' in request.form.keys():
        return redirect(url_for('first_questions', usr_id = 'user1'))

    if 'questions1' in request.form.keys():
        return redirect(url_for('room_picker', room = 'room1'))

    if 'end_rooms' in request.form.keys():
        return redirect(url_for('second_questions', usr_id = 'user1'))

    if 'questions2' in request.form.keys():
        return redirect(url_for('demographics' , usr_id = 'user1'))

    if 'demographics' in request.form.keys():
        return redirect(url_for('payment'))

    if 'payment' in request.form.keys():
        return redirect(url_for('thanks'))


# instructions page, with a short written answer section, saved to database #
@app.route('/i/<path:usr_id>') #2
def instructions(usr_id):
    return render_template('instructions.html')

# one page form w/ questions, answers written to database #
@app.route('/q1/<path:usr_id>') #3
def first_questions(usr_id):
    return render_template('questions1.html')

# build the rooms from unique identifiers #
@app.route('/<path:room>') #4
def room_picker(room):
    context = {'room': room} #pass args to the template renderer for the specific room
    return render_template('room.html', **context)

# one page form w/ questions, answers written to database #
@app.route('/q2/<path:usr_id>') #5
def second_questions(usr_id):
    return render_template('questions2.html')

@app.route('/demos/<path:usr_id>')
def demographics(usr_id):
    return render_template('demos.html')

@app.route('/checkout/<path:usr_id>')
def payment(usr_id):
    return render_template('payment.html')

@app.route('/ty/<path:usr_id>')
def thanks(usr_id):
    return render_template('thanks.html')


# helpers #

#need to keep chat namespace dicts in order
#easiest way to key is by id of namespace instance
#

#join or create a room
def join_or_create():
    pass
    



# namespace #

class ChatNamespace(BaseNamespace, RoomsMixin):
    '''
    Namespace for socket.io chatroom interaction.
    Broadcast only to your room.
    Interactions are stored appropriately in the database.
    Any information accessible to all instances of the namespace should be stored on self.session
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

    '''def on_username(self, usr):
        room = 'one' #temp
        print 'username', self
        self.session['name'] = usr
        self.session_by_id[id(self)]['user'] = usr
        self.session_by_id[id(self)]['room'] = str(room)
        self.room_dict[room] = usr #put individual in a room
        self.emit("room_url", url_for('room_picker', room = room))'''

    def on_join(self, room):
        print 'joined'
        self.join(str(room))

    def on_user_message(self, msg):
        #assumes a max of 1 room
        print 'message away!'
        self.emit_to_room('one', 'msg_to_room', users[0], msg)


# socket io server always defaults here #

@app.route('/socket.io/<path:rest>')
def push_stream(rest):
    try:
        real_request = request._get_current_object()
        socketio_manage(request.environ, {'/chat': ChatNamespace}, request=real_request)
    except:
        app.logger.error('Exception while handling sockcketio connection', exc_info=True)
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