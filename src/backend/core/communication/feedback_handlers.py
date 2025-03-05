"""
Feedback Handlers

This module provides handlers for feedback-related operations, including
submission, retrieval, and analysis of user feedback.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from ...storage.database.feedback_manager import FeedbackManager

# Configure logging
logger = logging.getLogger(__name__)

# Initialize feedback manager
feedback_manager = FeedbackManager()

# Forward declarations of handlers
def handle_get_feedback_categories(data):
    pass

def handle_submit_feedback(data):
    pass

def handle_get_feedback(data):
    pass

def handle_search_feedback(data):
    pass

def handle_add_feedback_response(data):
    pass

def handle_update_feature_request(data):
    pass

def handle_get_feedback_analytics(data):
    pass

# Dictionary of handlers for feedback operations
FEEDBACK_HANDLERS = {
    "get_feedback_categories": handle_get_feedback_categories,
    "submit_feedback": handle_submit_feedback,
    "get_feedback": handle_get_feedback,
    "search_feedback": handle_search_feedback,
    "add_feedback_response": handle_add_feedback_response,
    "update_feature_request": handle_update_feature_request,
    "get_feedback_analytics": handle_get_feedback_analytics
}

def handle_get_feedback_categories(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle request to get all feedback categories.
    
    Args:
        data: Request data (empty for this handler)
        
    Returns:
        Response with list of categories
    """
    try:
        categories = feedback_manager.get_categories()
        return {
            "success": True,
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Error getting feedback categories: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def handle_submit_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle request to submit feedback.
    
    Args:
        data: Request data with feedback information
        
    Returns:
        Response with success status and feedback ID
    """
    try:
        # Extract required parameters
        title = data.get("title")
        description = data.get("description", "")
        category_id = data.get("category_id")
        rating = data.get("rating")
        
        # Extract optional parameters
        user_id = data.get("user_id")
        is_anonymous = data.get("is_anonymous", False)
        context = data.get("context")
        tags = data.get("tags")
        feature_request = data.get("feature_request")
        
        # Validate required parameters
        if not title:
            return {
                "success": False,
                "error": "Title is required"
            }
        
        if not category_id:
            return {
                "success": False,
                "error": "Category ID is required"
            }
        
        # Submit feedback
        feedback_id = feedback_manager.submit_feedback(
            title=title,
            description=description,
            category_id=category_id,
            rating=rating,
            user_id=user_id,
            is_anonymous=is_anonymous,
            context=context,
            tags=tags,
            feature_request=feature_request
        )
        
        return {
            "success": True,
            "feedback_id": feedback_id
        }
    except ValueError as e:
        logger.error(f"Validation error in submit_feedback: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return {
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }

def handle_get_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle request to get feedback by ID.
    
    Args:
        data: Request data with feedback ID
        
    Returns:
        Response with feedback details
    """
    try:
        # Extract parameters
        feedback_id = data.get("feedback_id")
        
        # Validate parameters
        if not feedback_id:
            return {
                "success": False,
                "error": "Feedback ID is required"
            }
        
        # Get feedback
        feedback = feedback_manager.get_feedback(feedback_id)
        
        if not feedback:
            return {
                "success": False,
                "error": f"Feedback with ID {feedback_id} not found"
            }
        
        return {
            "success": True,
            "feedback": feedback
        }
    except Exception as e:
        logger.error(f"Error getting feedback: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def handle_search_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle request to search for feedback.
    
    Args:
        data: Request data with search criteria
        
    Returns:
        Response with search results
    """
    try:
        # Extract parameters
        category_id = data.get("category_id")
        min_rating = data.get("min_rating")
        max_rating = data.get("max_rating")
        tags = data.get("tags")
        context_type = data.get("context_type")
        context_id = data.get("context_id")
        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")
        limit = data.get("limit", 100)
        offset = data.get("offset", 0)
        
        # Parse dates if provided
        start_date = None
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str)
        
        end_date = None
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
        
        # Search feedback
        results, total_count = feedback_manager.search_feedback(
            category_id=category_id,
            min_rating=min_rating,
            max_rating=max_rating,
            tags=tags,
            context_type=context_type,
            context_id=context_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "results": results,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error searching feedback: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def handle_add_feedback_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle request to add a response to feedback.
    
    Args:
        data: Request data with response information
        
    Returns:
        Response with success status and response ID
    """
    try:
        # Extract parameters
        feedback_id = data.get("feedback_id")
        response_text = data.get("response_text")
        responded_by = data.get("responded_by")
        is_public = data.get("is_public", True)
        
        # Validate parameters
        if not feedback_id:
            return {
                "success": False,
                "error": "Feedback ID is required"
            }
        
        if not response_text:
            return {
                "success": False,
                "error": "Response text is required"
            }
        
        # Add response
        response_id = feedback_manager.add_response(
            feedback_id=feedback_id,
            response_text=response_text,
            responded_by=responded_by,
            is_public=is_public
        )
        
        return {
            "success": True,
            "response_id": response_id
        }
    except ValueError as e:
        logger.error(f"Validation error in add_feedback_response: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Error adding feedback response: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def handle_update_feature_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle request to update a feature request.
    
    Args:
        data: Request data with feature request information
        
    Returns:
        Response with success status
    """
    try:
        # Extract parameters
        feedback_id = data.get("feedback_id")
        status = data.get("status")
        priority = data.get("priority")
        votes = data.get("votes")
        assigned_to = data.get("assigned_to")
        planned_release = data.get("planned_release")
        notes = data.get("notes")
        
        # Validate parameters
        if not feedback_id:
            return {
                "success": False,
                "error": "Feedback ID is required"
            }
        
        # Update feature request
        success = feedback_manager.update_feature_request(
            feedback_id=feedback_id,
            status=status,
            priority=priority,
            votes=votes,
            assigned_to=assigned_to,
            planned_release=planned_release,
            notes=notes
        )
        
        if not success:
            return {
                "success": False,
                "error": "Feature request not found"
            }
        
        return {
            "success": True
        }
    except ValueError as e:
        logger.error(f"Validation error in update_feature_request: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Error updating feature request: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def handle_get_feedback_analytics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle request to get feedback analytics.
    
    Args:
        data: Request data with analytics criteria
        
    Returns:
        Response with analytics information
    """
    try:
        # Extract parameters
        category_id = data.get("category_id")
        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")
        
        # Parse dates if provided
        start_date = None
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str)
        
        end_date = None
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
        
        # Get analytics
        analytics = feedback_manager.get_analytics(
            category_id=category_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "success": True,
            "analytics": analytics
        }
    except Exception as e:
        logger.error(f"Error getting feedback analytics: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }