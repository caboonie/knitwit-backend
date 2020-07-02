from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from flask_login import UserMixin
from passlib.apps import custom_app_context as pwd_security

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    password_hash = Column(String)
    username = Column(String)
    token = Column(String)

    def hash_password(self, password):
        self.password_hash = pwd_security.encrypt(password)
    def verify_password(self, password):
        return pwd_security.verify(password, self.password_hash)

class Pattern(Base):
    __tablename__ = "patterns"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    pattern_json = Column(String)
    name = Column(String)
    timestamp = Column(String)