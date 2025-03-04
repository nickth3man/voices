/**
 * Feedback Form Component
 * 
 * This component provides a form for users to submit feedback about the application,
 * including ratings, comments, and feature suggestions.
 */

import React, { useState, useEffect } from 'react';
import PythonBridge from '../../controllers/PythonBridge';

const FeedbackForm = ({ onSubmitSuccess }) => {
  // State for form fields
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [categoryId, setCategoryId] = useState('');
  const [rating, setRating] = useState(0);
  const [isAnonymous, setIsAnonymous] = useState(false);
  const [tags, setTags] = useState([]);
  const [tagInput, setTagInput] = useState('');
  const [isFeatureRequest, setIsFeatureRequest] = useState(false);
  const [featureRequestPriority, setFeatureRequestPriority] = useState('medium');
  
  // State for categories
  const [categories, setCategories] = useState([]);
  
  // State for loading and error handling
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  
  // Load categories on component mount
  useEffect(() => {
    loadCategories();
  }, []);
  
  // Load categories from backend
  const loadCategories = async () => {
    try {
      const response = await PythonBridge.sendCommand('get_feedback_categories', {});
      
      if (response.success) {
        setCategories(response.categories);
        
        // Set default category if available
        if (response.categories.length > 0) {
          setCategoryId(response.categories[0].id);
        }
      } else {
        setError('Failed to load categories: ' + response.error);
      }
    } catch (err) {
      setError('Error loading categories: ' + err.message);
    }
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate form
    if (!title) {
      setError('Please enter a title');
      return;
    }
    
    if (!categoryId) {
      setError('Please select a category');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Prepare feature request data if applicable
      const featureRequest = isFeatureRequest ? {
        status: 'new',
        priority: featureRequestPriority
      } : null;
      
      // Submit feedback
      const response = await PythonBridge.sendCommand('submit_feedback', {
        title,
        description,
        category_id: parseInt(categoryId),
        rating: rating > 0 ? rating : null,
        is_anonymous: isAnonymous,
        tags: tags.length > 0 ? tags : null,
        feature_request: featureRequest
      });
      
      if (response.success) {
        setSuccess(true);
        
        // Reset form
        setTitle('');
        setDescription('');
        setRating(0);
        setIsAnonymous(false);
        setTags([]);
        setTagInput('');
        setIsFeatureRequest(false);
        setFeatureRequestPriority('medium');
        
        // Call success callback if provided
        if (onSubmitSuccess) {
          onSubmitSuccess(response.feedback_id);
        }
        
        // Clear success message after 3 seconds
        setTimeout(() => {
          setSuccess(false);
        }, 3000);
      } else {
        setError('Failed to submit feedback: ' + response.error);
      }
    } catch (err) {
      setError('Error submitting feedback: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle tag input
  const handleTagInputKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      
      const newTag = tagInput.trim();
      
      if (newTag && !tags.includes(newTag)) {
        setTags([...tags, newTag]);
        setTagInput('');
      }
    }
  };
  
  // Remove tag
  const removeTag = (tagToRemove) => {
    setTags(tags.filter(tag => tag !== tagToRemove));
  };
  
  // Render star rating
  const renderStarRating = () => {
    const stars = [];
    
    for (let i = 1; i <= 5; i++) {
      stars.push(
        <span
          key={i}
          className={`rating-star ${i <= rating ? 'selected' : ''}`}
          onClick={() => setRating(i)}
        >
          ★
        </span>
      );
    }
    
    return (
      <div className="rating-container">
        <label>Rating</label>
        <div className="rating-stars">
          {stars}
        </div>
        <div className="rating-labels">
          <span>Poor</span>
          <span>Excellent</span>
        </div>
      </div>
    );
  };
  
  return (
    <div className="feedback-section">
      <h3>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
          <path d="M2 1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h9.586a2 2 0 0 1 1.414.586l2 2V2a1 1 0 0 0-1-1H2zm12-1a2 2 0 0 1 2 2v12.793a.5.5 0 0 1-.854.353l-2.853-2.853a1 1 0 0 0-.707-.293H2a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2h12z"/>
          <path d="M3 3.5a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9a.5.5 0 0 1-.5-.5zM3 6a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9A.5.5 0 0 1 3 6zm0 2.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5z"/>
        </svg>
        Submit Feedback
      </h3>
      <p className="section-description">
        We value your feedback! Please share your thoughts, suggestions, or report any issues you've encountered.
      </p>
      
      {error && (
        <div className="error-message" style={{ color: 'var(--error-color)', marginBottom: 'var(--spacing-md)' }}>
          {error}
        </div>
      )}
      
      {success && (
        <div className="success-message" style={{ color: 'var(--success-color)', marginBottom: 'var(--spacing-md)' }}>
          Thank you for your feedback!
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="feedback-title">Title</label>
          <input
            type="text"
            id="feedback-title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Brief summary of your feedback"
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="feedback-category">Category</label>
          <select
            id="feedback-category"
            value={categoryId}
            onChange={(e) => setCategoryId(e.target.value)}
            required
          >
            <option value="">Select a category</option>
            {categories.map(category => (
              <option key={category.id} value={category.id}>
                {category.name}
              </option>
            ))}
          </select>
        </div>
        
        {renderStarRating()}
        
        <div className="form-group">
          <label htmlFor="feedback-description">Description</label>
          <textarea
            id="feedback-description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Please provide details about your feedback"
          />
        </div>
        
        <div className="tags-input">
          <label>Tags</label>
          <div className="tags-container">
            {tags.map(tag => (
              <div key={tag} className="tag">
                {tag}
                <span className="remove-tag" onClick={() => removeTag(tag)}>×</span>
              </div>
            ))}
            <input
              type="text"
              className="tags-input-field"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={handleTagInputKeyDown}
              placeholder="Add tags (press Enter or comma to add)"
            />
          </div>
        </div>
        
        <div className="form-group">
          <label>
            <input
              type="checkbox"
              checked={isFeatureRequest}
              onChange={(e) => setIsFeatureRequest(e.target.checked)}
            />
            This is a feature request
          </label>
        </div>
        
        {isFeatureRequest && (
          <div className="form-group">
            <label htmlFor="feature-priority">Priority</label>
            <select
              id="feature-priority"
              value={featureRequestPriority}
              onChange={(e) => setFeatureRequestPriority(e.target.value)}
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </select>
          </div>
        )}
        
        <div className="anonymous-toggle">
          <input
            type="checkbox"
            id="anonymous-feedback"
            checked={isAnonymous}
            onChange={(e) => setIsAnonymous(e.target.checked)}
          />
          <label htmlFor="anonymous-feedback">Submit anonymously</label>
        </div>
        
        <button
          type="submit"
          className="submit-button"
          disabled={loading}
        >
          {loading ? 'Submitting...' : 'Submit Feedback'}
        </button>
      </form>
    </div>
  );
};

export default FeedbackForm;