from sqlalchemy import (
    Column, String, DateTime, Integer, Text, Date, Boolean, ForeignKey,Text,
    PrimaryKeyConstraint, ARRAY, UUID, Index
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
import uuid
from sqlalchemy.sql.schema import Computed

Base = declarative_base()

class Channels(Base):
    __tablename__ = 'channels'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    subscribers = Column(Integer, nullable=True)
    url = Column(Text, nullable=False)
    types_reactions = Column(ARRAY(Text), nullable=True)
    media = Column(Boolean, default=False, nullable=True)
    lang = Column(String(2), nullable=True)
    date_begin = Column(Date, nullable=True)
    used = Column(Boolean, default=False, nullable=False)
    processing_depth = Column(Integer, default = 5000, nullable=False)
    comment = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Channel(id={self.id}, name='{self.name}')>"

class Narratives(Base):
    __tablename__ = 'narratives'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(Integer, nullable=False)
    date = Column(DateTime, nullable=False)
    message = Column(Text, nullable=False)
    id_channel = Column(Integer, ForeignKey('channels.id'), nullable=False)
    message_vector = Column(TSVECTOR, Computed("to_tsvector('russian', message)", persisted=True))

    __table_args__ = (
        Index('idx_narratives_message_search', 'message_vector', postgresql_using='gin'),
    )

    def __repr__(self):
        return f"<Narrative(id={self.id}, message_id={self.message_id})>"

class MessagesDay(Base):
    __tablename__ = 'message_day'

    id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    id_channel = Column(Integer, ForeignKey('channels.id'), nullable=False)
    cur_date = Column(Date, nullable=False)
    message_id = Column(UUID, nullable=False)
    views = Column(Integer, nullable=False)
    forwards = Column(Integer, nullable=False)
    replies = Column(Integer, nullable=False)
    reactions = Column(Integer, nullable=False)
    delta_views = Column(Integer, nullable=False)
    delta_forwards = Column(Integer, nullable=False)
    delta_replies = Column(Integer, nullable=False)
    delta_reactions = Column(Integer, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('id', 'cur_date'),
        {
            'postgresql_partition_by': 'RANGE (cur_date)',
            'schema': 'public'
        }
    )

    def __repr__(self):
        return f"<MessageDay(id={self.id}, date={self.cur_date})>"