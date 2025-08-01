from traceback import extract_stack
from langchain_core import messages
from requests import session
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, create_engine, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("ChatMessage", back_populates="session")
    files = relationship("UploadedFile", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    sender = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")

class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    filename = Column(String)
    extracted_text = Column(Text)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="files")

#Setup engine and session[Database and Tables]
engine = create_engine("sqlite:///chat_history.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

