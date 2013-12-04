'''
@by George Berry (geb97@cornell.edu)
@Cornell Dpt of Sociology (Social Dynamics Lab)
@December 2013
'''

from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

class ChatRoom(db.Model):
    __tablename__ = 'chatrooms'
    id = db.Column(db.Integer, primary_key=True)
    room_name = db.Column(db.String(8), nullable=False)
    text = db.Column(db.String(1000))
    users = db.relationship('User', backref='chatrooms', lazy='dynamic') #allows getting chatrooms by calling User.chatrooms


class User(db.Model):
    __tablename__ = 'chatusers'
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(6), nullable=False)
    chatroom_id = db.Column(db.String(8), db.ForeignKey('chatrooms.id'))
    
