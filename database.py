from models import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

engine = create_engine('sqlite:///users.db?check_same_thread=False')
Base.metadata.create_all(engine)
DBSession = sessionmaker(bind=engine)
session = DBSession()

def create_user(name,secret_word):
    user = User(username=name)
    user.hash_password(secret_word)
    session.add(user)
    session.commit()
    return user

def get_user(username):
    return session.query(User).filter_by(username=username).first()

def get_user_id(id_num):
    return session.query(User).filter_by(id=id_num).first()

# TODO - salt and hash the tokens
def set_user_token(id_num, token):
    user = session.query(User).filter_by(id=id_num).first()
    user.token = token
    session.add(user)
    session.commit()

def check_token(id_num, token):
    user = session.query(User).filter_by(id=id_num).first()
    return user.token == token


def add_pattern(user_id, name, pattern_json, timestamp=datetime.now().strftime("%m/%d/%Y, %H:%M:%S")):
    pattern = Pattern(user_id = user_id, name=name, pattern_json = pattern_json, timestamp = timestamp)
    session.add(pattern)
    session.commit()
    return pattern

def get_pattern(id_num):
    return session.query(Pattern).filter_by(id=id_num).first()

def get_users_patterns(user_id):
    return session.query(Pattern).filter_by(user_id=user_id).all()

def get_pattern_user_name(user, name):
    return session.query(Pattern).filter_by(user_id=user.id).filter_by(name=name).first()

def update_pattern(id_num, pattern_json, timestamp=None):
    if timestamp == None:
        timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print("UPDATING",timestamp)
    pattern = session.query(Pattern).filter_by(id=id_num).first()
    pattern.pattern_json = pattern_json
    pattern.timestamp = timestamp
    session.add(pattern)
    session.commit()

