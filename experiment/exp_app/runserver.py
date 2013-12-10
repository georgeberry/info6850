'''
@by George Berry (geb97@cornell.edu)
@Cornell Dpt of Sociology (Social Dynamics Lab)
@December 2013
'''
from flask import Flask


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////db/chat.db'
db = SQLAlchemy(app)

import exp_app.views