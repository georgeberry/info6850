'''
@by George Berry (geb97@cornell.edu)
@Cornell Dpt of Sociology (Social Dynamics Lab)
@December 2013
'''

from rooms.py import app

def run_server():
    port = 6020
    SocketIOServer(('', port), app, resource='socket.io').serve_forever()