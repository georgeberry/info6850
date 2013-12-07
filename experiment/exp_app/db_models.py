'''
@by George Berry (geb97@cornell.edu)
@Cornell Dpt of Sociology (Social Dynamics Lab)
@December 2013
'''

from flask.ext.sqlalchemy import SQLAlchemy

class ChatRoom(db.Model):
    __tablename__ = 'chatrooms'
    room_name = db.Column(db.String(8), nullable=False, primary_key=True)
    user_one = db.Column(db.String(6), nullable=False)
    user_two = db.Column(db.String(6), nullable=False)
    user_three = db.Column(db.String(6), nullable=False)
    conversation = db.Column(text, nullable=False)
    
    users = db.relationship('User', backref='chatrooms', lazy='dynamic') #allows getting chatrooms by calling User.chatrooms


class User(db.Model):
    __tablename__ = 'chatusers'
    user_name = db.Column(db.String(6), nullable=False, primary_key=True)
    chatroom_id = db.Column(db.String(8), db.ForeignKey('chatrooms.id'))
    
