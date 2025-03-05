"""
Feedback Models

This module defines SQLAlchemy ORM models for the User Feedback Collection System.
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, Float, ForeignKey, Date, DateTime, func
from sqlalchemy.orm import relationship
from .models import Base

class FeedbackCategory(Base):
    """Model for feedback categories."""
    __tablename__ = 'feedback_categories'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    feedback = relationship("Feedback", back_populates="category")
    analytics = relationship("FeedbackAnalytics", back_populates="category")

    def __repr__(self):
        return f"<FeedbackCategory(id={self.id}, name='{self.name}')>"


class Feedback(Base):
    """Model for user feedback."""
    __tablename__ = 'feedback'

    id = Column(Integer, primary_key=True)
    user_id = Column(String)  # Optional, can be NULL for anonymous feedback
    category_id = Column(Integer, ForeignKey('feedback_categories.id'))
    title = Column(String, nullable=False)
    description = Column(Text)
    rating = Column(Integer)  # 1-5 star rating
    is_anonymous = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    category = relationship("FeedbackCategory", back_populates="feedback")
    context = relationship("FeedbackContext", back_populates="feedback", cascade="all, delete-orphan")
    tags = relationship("FeedbackTag", secondary="feedback_tag_mapping", back_populates="feedback")
    feature_request = relationship("FeedbackFeatureRequest", back_populates="feedback", uselist=False, cascade="all, delete-orphan")
    responses = relationship("FeedbackResponse", back_populates="feedback", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Feedback(id={self.id}, title='{self.title}', rating={self.rating})>"


class FeedbackContext(Base):
    """Model for feedback context information."""
    __tablename__ = 'feedback_context'

    id = Column(Integer, primary_key=True)
    feedback_id = Column(Integer, ForeignKey('feedback.id', ondelete='CASCADE'), nullable=False)
    context_type = Column(String, nullable=False)  # 'feature', 'model', 'file', etc.
    context_id = Column(String)  # ID of the related entity (feature name, model ID, file ID, etc.)
    context_data = Column(Text)  # JSON data with additional context information
    created_at = Column(DateTime, default=func.current_timestamp())

    # Relationships
    feedback = relationship("Feedback", back_populates="context")

    def __repr__(self):
        return f"<FeedbackContext(id={self.id}, type='{self.context_type}', context_id='{self.context_id}')>"


class FeedbackTag(Base):
    """Model for feedback tags."""
    __tablename__ = 'feedback_tags'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, default=func.current_timestamp())

    # Relationships
    feedback = relationship("Feedback", secondary="feedback_tag_mapping", back_populates="tags")

    def __repr__(self):
        return f"<FeedbackTag(id={self.id}, name='{self.name}')>"


class FeedbackTagMapping(Base):
    """Model for mapping between feedback and tags (many-to-many relationship)."""
    __tablename__ = 'feedback_tag_mapping'

    feedback_id = Column(Integer, ForeignKey('feedback.id', ondelete='CASCADE'), primary_key=True)
    tag_id = Column(Integer, ForeignKey('feedback_tags.id', ondelete='CASCADE'), primary_key=True)
    created_at = Column(DateTime, default=func.current_timestamp())


class FeedbackAnalytics(Base):
    """Model for aggregated feedback analytics."""
    __tablename__ = 'feedback_analytics'

    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey('feedback_categories.id'))
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    count = Column(Integer, default=0, nullable=False)
    average_rating = Column(Float)
    positive_count = Column(Integer, default=0)  # Ratings 4-5
    neutral_count = Column(Integer, default=0)   # Rating 3
    negative_count = Column(Integer, default=0)  # Ratings 1-2
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    category = relationship("FeedbackCategory", back_populates="analytics")

    def __repr__(self):
        return f"<FeedbackAnalytics(id={self.id}, period='{self.period_start} to {self.period_end}', count={self.count})>"


class FeedbackFeatureRequest(Base):
    """Model for feature requests derived from feedback."""
    __tablename__ = 'feedback_feature_requests'

    id = Column(Integer, primary_key=True)
    feedback_id = Column(Integer, ForeignKey('feedback.id', ondelete='CASCADE'), nullable=False)
    status = Column(String, default='new')  # 'new', 'under_review', 'planned', 'in_progress', 'completed', 'declined'
    priority = Column(String, default='medium')  # 'low', 'medium', 'high', 'critical'
    votes = Column(Integer, default=0)
    assigned_to = Column(String)
    planned_release = Column(String)
    notes = Column(Text)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    feedback = relationship("Feedback", back_populates="feature_request")

    def __repr__(self):
        return f"<FeedbackFeatureRequest(id={self.id}, status='{self.status}', priority='{self.priority}')>"


class FeedbackResponse(Base):
    """Model for responses to feedback."""
    __tablename__ = 'feedback_responses'

    id = Column(Integer, primary_key=True)
    feedback_id = Column(Integer, ForeignKey('feedback.id', ondelete='CASCADE'), nullable=False)
    response_text = Column(Text, nullable=False)
    responded_by = Column(String)
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    feedback = relationship("Feedback", back_populates="responses")

    def __repr__(self):
        return f"<FeedbackResponse(id={self.id}, is_public={self.is_public})>"