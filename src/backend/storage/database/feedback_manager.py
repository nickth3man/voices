"""
Feedback Manager

This module provides a manager class for handling feedback operations, including
submission, retrieval, analysis, and management of user feedback.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_

from .db_manager import DatabaseManager
from .feedback_models import (
    Feedback, FeedbackCategory, FeedbackContext, FeedbackTag,
    FeedbackTagMapping, FeedbackAnalytics, FeedbackFeatureRequest,
    FeedbackResponse
)

# Configure logging
logger = logging.getLogger(__name__)

class FeedbackManager:
    """Manager class for feedback operations."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the FeedbackManager.
        
        Args:
            db_manager: Optional DatabaseManager instance. If not provided, a new one will be created.
        """
        self.db_manager = db_manager or DatabaseManager()
        
    def get_categories(self, session: Optional[Session] = None) -> List[Dict[str, Any]]:
        """
        Get all feedback categories.
        
        Args:
            session: Optional SQLAlchemy session. If not provided, a new one will be created.
            
        Returns:
            List of dictionaries containing category information.
        """
        with self.db_manager.session_scope(session) as session:
            categories = session.query(FeedbackCategory).all()
            return [
                {
                    'id': category.id,
                    'name': category.name,
                    'description': category.description
                }
                for category in categories
            ]
    
    def submit_feedback(
        self,
        title: str,
        description: str,
        category_id: int,
        rating: int,
        user_id: Optional[str] = None,
        is_anonymous: bool = False,
        context: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        feature_request: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None
    ) -> int:
        """
        Submit new feedback.
        
        Args:
            title: Feedback title
            description: Feedback description
            category_id: Category ID
            rating: Rating (1-5)
            user_id: Optional user ID
            is_anonymous: Whether the feedback is anonymous
            context: Optional list of context dictionaries
            tags: Optional list of tag names
            feature_request: Optional feature request information
            session: Optional SQLAlchemy session
            
        Returns:
            ID of the created feedback
            
        Raises:
            ValueError: If required parameters are invalid
        """
        if not title:
            raise ValueError("Title is required")
        
        if rating and (rating < 1 or rating > 5):
            raise ValueError("Rating must be between 1 and 5")
        
        with self.db_manager.session_scope(session) as session:
            # Create feedback
            feedback = Feedback(
                title=title,
                description=description,
                category_id=category_id,
                rating=rating,
                user_id=user_id if not is_anonymous else None,
                is_anonymous=is_anonymous
            )
            session.add(feedback)
            session.flush()  # Flush to get the ID
            
            # Add context if provided
            if context:
                for ctx in context:
                    feedback_context = FeedbackContext(
                        feedback_id=feedback.id,
                        context_type=ctx.get('type'),
                        context_id=ctx.get('id'),
                        context_data=json.dumps(ctx.get('data', {}))
                    )
                    session.add(feedback_context)
            
            # Add tags if provided
            if tags:
                for tag_name in tags:
                    # Get or create tag
                    tag = session.query(FeedbackTag).filter(FeedbackTag.name == tag_name).first()
                    if not tag:
                        tag = FeedbackTag(name=tag_name)
                        session.add(tag)
                        session.flush()
                    
                    # Create mapping
                    mapping = FeedbackTagMapping(feedback_id=feedback.id, tag_id=tag.id)
                    session.add(mapping)
            
            # Add feature request if provided
            if feature_request:
                fr = FeedbackFeatureRequest(
                    feedback_id=feedback.id,
                    status=feature_request.get('status', 'new'),
                    priority=feature_request.get('priority', 'medium'),
                    notes=feature_request.get('notes')
                )
                session.add(fr)
            
            # Update analytics
            self._update_analytics(session, feedback)
            
            return feedback.id
    
    def get_feedback(
        self,
        feedback_id: int,
        session: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get feedback by ID.
        
        Args:
            feedback_id: Feedback ID
            session: Optional SQLAlchemy session
            
        Returns:
            Dictionary containing feedback information or None if not found
        """
        with self.db_manager.session_scope(session) as session:
            feedback = session.query(Feedback).filter(Feedback.id == feedback_id).first()
            
            if not feedback:
                return None
            
            # Get category
            category = feedback.category.name if feedback.category else None
            
            # Get context
            context = [
                {
                    'type': ctx.context_type,
                    'id': ctx.context_id,
                    'data': json.loads(ctx.context_data) if ctx.context_data else {}
                }
                for ctx in feedback.context
            ]
            
            # Get tags
            tags = [tag.name for tag in feedback.tags]
            
            # Get feature request
            feature_request = None
            if feedback.feature_request:
                fr = feedback.feature_request
                feature_request = {
                    'status': fr.status,
                    'priority': fr.priority,
                    'votes': fr.votes,
                    'assigned_to': fr.assigned_to,
                    'planned_release': fr.planned_release,
                    'notes': fr.notes
                }
            
            # Get responses
            responses = [
                {
                    'id': resp.id,
                    'text': resp.response_text,
                    'responded_by': resp.responded_by,
                    'is_public': resp.is_public,
                    'created_at': resp.created_at.isoformat()
                }
                for resp in feedback.responses if resp.is_public or not resp.is_public
            ]
            
            return {
                'id': feedback.id,
                'title': feedback.title,
                'description': feedback.description,
                'category': category,
                'rating': feedback.rating,
                'user_id': feedback.user_id,
                'is_anonymous': feedback.is_anonymous,
                'created_at': feedback.created_at.isoformat(),
                'context': context,
                'tags': tags,
                'feature_request': feature_request,
                'responses': responses
            }
    
    def search_feedback(
        self,
        category_id: Optional[int] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        tags: Optional[List[str]] = None,
        context_type: Optional[str] = None,
        context_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
        session: Optional[Session] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search for feedback based on various criteria.
        
        Args:
            category_id: Optional category ID filter
            min_rating: Optional minimum rating filter
            max_rating: Optional maximum rating filter
            tags: Optional list of tag names to filter by
            context_type: Optional context type filter
            context_id: Optional context ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results to return
            offset: Offset for pagination
            session: Optional SQLAlchemy session
            
        Returns:
            Tuple of (list of feedback dictionaries, total count)
        """
        with self.db_manager.session_scope(session) as session:
            # Build query
            query = session.query(Feedback)
            
            # Apply filters
            if category_id is not None:
                query = query.filter(Feedback.category_id == category_id)
            
            if min_rating is not None:
                query = query.filter(Feedback.rating >= min_rating)
            
            if max_rating is not None:
                query = query.filter(Feedback.rating <= max_rating)
            
            if tags:
                # Join with tags
                query = query.join(FeedbackTagMapping).join(FeedbackTag)
                query = query.filter(FeedbackTag.name.in_(tags))
            
            if context_type or context_id:
                # Join with context
                query = query.join(FeedbackContext)
                
                if context_type:
                    query = query.filter(FeedbackContext.context_type == context_type)
                
                if context_id:
                    query = query.filter(FeedbackContext.context_id == context_id)
            
            if start_date:
                query = query.filter(Feedback.created_at >= start_date)
            
            if end_date:
                query = query.filter(Feedback.created_at <= end_date)
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            query = query.order_by(desc(Feedback.created_at))
            query = query.limit(limit).offset(offset)
            
            # Execute query
            feedback_list = query.all()
            
            # Format results
            results = []
            for feedback in feedback_list:
                category = feedback.category.name if feedback.category else None
                tags = [tag.name for tag in feedback.tags]
                
                results.append({
                    'id': feedback.id,
                    'title': feedback.title,
                    'description': feedback.description,
                    'category': category,
                    'rating': feedback.rating,
                    'is_anonymous': feedback.is_anonymous,
                    'created_at': feedback.created_at.isoformat(),
                    'tags': tags
                })
            
            return results, total_count
    
    def add_response(
        self,
        feedback_id: int,
        response_text: str,
        responded_by: Optional[str] = None,
        is_public: bool = True,
        session: Optional[Session] = None
    ) -> int:
        """
        Add a response to feedback.
        
        Args:
            feedback_id: Feedback ID
            response_text: Response text
            responded_by: Optional responder identifier
            is_public: Whether the response is public
            session: Optional SQLAlchemy session
            
        Returns:
            ID of the created response
            
        Raises:
            ValueError: If feedback_id is invalid or response_text is empty
        """
        if not response_text:
            raise ValueError("Response text is required")
        
        with self.db_manager.session_scope(session) as session:
            # Check if feedback exists
            feedback = session.query(Feedback).filter(Feedback.id == feedback_id).first()
            if not feedback:
                raise ValueError(f"Feedback with ID {feedback_id} not found")
            
            # Create response
            response = FeedbackResponse(
                feedback_id=feedback_id,
                response_text=response_text,
                responded_by=responded_by,
                is_public=is_public
            )
            session.add(response)
            session.flush()
            
            return response.id
    
    def update_feature_request(
        self,
        feedback_id: int,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        votes: Optional[int] = None,
        assigned_to: Optional[str] = None,
        planned_release: Optional[str] = None,
        notes: Optional[str] = None,
        session: Optional[Session] = None
    ) -> bool:
        """
        Update a feature request.
        
        Args:
            feedback_id: Feedback ID
            status: Optional new status
            priority: Optional new priority
            votes: Optional new vote count
            assigned_to: Optional new assignee
            planned_release: Optional new planned release
            notes: Optional new notes
            session: Optional SQLAlchemy session
            
        Returns:
            True if successful, False if feature request not found
            
        Raises:
            ValueError: If feedback_id is invalid
        """
        with self.db_manager.session_scope(session) as session:
            # Check if feedback exists
            feedback = session.query(Feedback).filter(Feedback.id == feedback_id).first()
            if not feedback:
                raise ValueError(f"Feedback with ID {feedback_id} not found")
            
            # Get or create feature request
            feature_request = feedback.feature_request
            if not feature_request:
                feature_request = FeedbackFeatureRequest(feedback_id=feedback_id)
                session.add(feature_request)
            
            # Update fields
            if status is not None:
                feature_request.status = status
            
            if priority is not None:
                feature_request.priority = priority
            
            if votes is not None:
                feature_request.votes = votes
            
            if assigned_to is not None:
                feature_request.assigned_to = assigned_to
            
            if planned_release is not None:
                feature_request.planned_release = planned_release
            
            if notes is not None:
                feature_request.notes = notes
            
            return True
    
    def get_analytics(
        self,
        category_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Get feedback analytics.
        
        Args:
            category_id: Optional category ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            session: Optional SQLAlchemy session
            
        Returns:
            Dictionary containing analytics information
        """
        with self.db_manager.session_scope(session) as session:
            # Default date range is last 30 days
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            
            if not end_date:
                end_date = datetime.now()
            
            # Build query
            query = session.query(
                func.count(Feedback.id).label('count'),
                func.avg(Feedback.rating).label('average_rating'),
                func.sum(case((Feedback.rating >= 4, 1), else_=0)).label('positive_count'),
                func.sum(case((Feedback.rating == 3, 1), else_=0)).label('neutral_count'),
                func.sum(case((Feedback.rating <= 2, 1), else_=0)).label('negative_count')
            )
            
            # Apply filters
            query = query.filter(Feedback.created_at >= start_date)
            query = query.filter(Feedback.created_at <= end_date)
            
            if category_id is not None:
                query = query.filter(Feedback.category_id == category_id)
            
            # Execute query
            result = query.first()
            
            # Get category breakdown
            category_query = session.query(
                FeedbackCategory.name,
                func.count(Feedback.id).label('count'),
                func.avg(Feedback.rating).label('average_rating')
            ).join(Feedback, FeedbackCategory.id == Feedback.category_id)
            
            # Apply date filters
            category_query = category_query.filter(Feedback.created_at >= start_date)
            category_query = category_query.filter(Feedback.created_at <= end_date)
            
            # Group by category
            category_query = category_query.group_by(FeedbackCategory.name)
            
            # Execute query
            category_results = category_query.all()
            
            # Format category results
            categories = [
                {
                    'name': name,
                    'count': count,
                    'average_rating': float(avg_rating) if avg_rating else None
                }
                for name, count, avg_rating in category_results
            ]
            
            # Get tag breakdown
            tag_query = session.query(
                FeedbackTag.name,
                func.count(Feedback.id).label('count')
            ).join(FeedbackTagMapping, FeedbackTag.id == FeedbackTagMapping.tag_id
            ).join(Feedback, FeedbackTagMapping.feedback_id == Feedback.id)
            
            # Apply date filters
            tag_query = tag_query.filter(Feedback.created_at >= start_date)
            tag_query = tag_query.filter(Feedback.created_at <= end_date)
            
            # Apply category filter if provided
            if category_id is not None:
                tag_query = tag_query.filter(Feedback.category_id == category_id)
            
            # Group by tag
            tag_query = tag_query.group_by(FeedbackTag.name)
            
            # Order by count descending
            tag_query = tag_query.order_by(desc('count'))
            
            # Limit to top 10
            tag_query = tag_query.limit(10)
            
            # Execute query
            tag_results = tag_query.all()
            
            # Format tag results
            tags = [
                {
                    'name': name,
                    'count': count
                }
                for name, count in tag_results
            ]
            
            return {
                'total_count': result.count if result.count else 0,
                'average_rating': float(result.average_rating) if result.average_rating else None,
                'positive_count': result.positive_count if result.positive_count else 0,
                'neutral_count': result.neutral_count if result.neutral_count else 0,
                'negative_count': result.negative_count if result.negative_count else 0,
                'categories': categories,
                'top_tags': tags,
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            }
    
    def _update_analytics(self, session: Session, feedback: Feedback) -> None:
        """
        Update analytics after feedback submission.
        
        Args:
            session: SQLAlchemy session
            feedback: Feedback instance
        """
        # Get current date
        today = datetime.now().date()
        
        # Check if analytics entry exists for today
        analytics = session.query(FeedbackAnalytics).filter(
            FeedbackAnalytics.category_id == feedback.category_id,
            FeedbackAnalytics.period_start == today,
            FeedbackAnalytics.period_end == today
        ).first()
        
        if not analytics:
            # Create new analytics entry
            analytics = FeedbackAnalytics(
                category_id=feedback.category_id,
                period_start=today,
                period_end=today,
                count=1,
                average_rating=feedback.rating,
                positive_count=1 if feedback.rating >= 4 else 0,
                neutral_count=1 if feedback.rating == 3 else 0,
                negative_count=1 if feedback.rating <= 2 else 0
            )
            session.add(analytics)
        else:
            # Update existing analytics entry
            analytics.count += 1
            
            # Update rating counts
            if feedback.rating >= 4:
                analytics.positive_count += 1
            elif feedback.rating == 3:
                analytics.neutral_count += 1
            elif feedback.rating <= 2:
                analytics.negative_count += 1
            
            # Update average rating
            if feedback.rating:
                total_rating = analytics.average_rating * (analytics.count - 1) + feedback.rating
                analytics.average_rating = total_rating / analytics.count