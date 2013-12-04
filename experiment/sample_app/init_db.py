from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
import random, string
from types import IntType
import cPickle as pickle

#generates num names and 2*num room names
#names are 6 digit hexidecimal, #room names are 8 digit hexidecimal
def gen_names(num, name_length, room_length):

    assert type(num) is IntType, 'requires integer'

    user_names = [''.join(random.choice(string.hexdigits).lower() for n in xrange(name_length)) for each in xrange(num)]
    unique_names = len(set(user_names))
    total_names = len(user_names)

    room_names = [''.join(random.choice(string.hexdigits).lower() for n in xrange(room_length)) for each in xrange(2*num)]
    unique_rooms = len(set(room_names))
    total_rooms = len(room_names)

    while (unique_names != total_names) and (unique_rooms != total_rooms):
        user_names = [''.join(random.choice(string.hexdigits).lower() for n in xrange(name_length)) for each in xrange(num)]
        unique_names = len(set(user_names))
        total_names = len(user_names)

        room_names = [''.join(random.choice(string.hexdigits).lower() for n in xrange(room_length)) for each in xrange(2*num)]
        unique_rooms = len(set(room_names))
        total_rooms = len(room_names)

    return dict.fromkeys(user_names, {}), dict.fromkeys(room_names, {})


#should have a check to ask the user if they want to overwrite names
pickle.dump(gen_names(500,6,8), open('db/names.p', 'wb'))