/**
 * Feedback List Component
 * 
 * This component displays a list of feedback items with filtering and pagination.
 */

import React, { useState, useEffect } from 'react';
import PythonBridge from '../../controllers/PythonBridge';

const FeedbackList = () => {
  // State for feedback items
  const [feedbackItems, setFeedbackItems] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  
  // State for pagination
  const [limit, setLimit] = useState(10);
  const [offset, setOffset] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  
  // State for filters
  const [categoryId, setCategoryId] = useState('');
  const [minRating, setMinRating] = useState('');
  const [maxRating, setMaxRating] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [searchTags, setSearchTags] = useState('');
  
  // State for categories
  const [categories, setCategories] = useState([]);
  
  // State for loading and error handling
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Load categories and feedback on component mount
  useEffect(() => {
    loadCategories();
    loadFeedback();
  }, []);
  
  // Load feedback when pagination or filters change
  useEffect(() => {
    loadFeedback();
  }, [currentPage, limit, categoryId, minRating, maxRating, startDate, endDate, searchTags]);
  
  // Load categories from backend
  const loadCategories = async () => {
    try {
      const response = await PythonBridge.sendCommand('get_feedback_categories', {});
      
      if (response.success) {
        setCategories(response.categories);
      } else {
        setError('Failed to load categories: ' + response.error);
      }
    } catch (err) {
      setError('Error loading categories: ' + err.message);
    }
  };
  
  // Load feedback from backend
  const loadFeedback = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Prepare filter parameters
      const params = {
        limit,
        offset: (currentPage - 1) * limit
      };
      
      if (categoryId) {
        params.category_id = parseInt(categoryId);
      }
      
      if (minRating) {
        params.min_rating = parseInt(minRating);
      }
      
      if (maxRating) {
        params.max_rating = parseInt(maxRating);
      }
      
      if (startDate) {
        params.start_date = new Date(startDate).toISOString();
      }
      
      if (endDate) {
        params.end_date = new Date(endDate).toISOString();
      }
      
      if (searchTags) {
        params.tags = searchTags.split(',').map(tag => tag.trim());
      }
      
      // Search feedback
      const response = await PythonBridge.sendCommand('search_feedback', params);
      
      if (response.success) {
        setFeedbackItems(response.results);
        setTotalCount(response.total_count);
      } else {
        setError('Failed to load feedback: ' + response.error);
      }
    } catch (err) {
      setError('Error loading feedback: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle page change
  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
  };
  
  // Handle filter change
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    
    switch (name) {
      case 'category':
        setCategoryId(value);
        break;
      case 'minRating':
        setMinRating(value);
        break;
      case 'maxRating':
        setMaxRating(value);
        break;
      case 'startDate':
        setStartDate(value);
        break;
      case 'endDate':
        setEndDate(value);
        break;
      case 'tags':
        setSearchTags(value);
        break;
      default:
        break;
    }
    
    // Reset to first page when filters change
    setCurrentPage(1);
  };
  
  // Reset filters
  const resetFilters = () => {
    setCategoryId('');
    setMinRating('');
    setMaxRating('');
    setStartDate('');
    setEndDate('');
    setSearchTags('');
    setCurrentPage(1);
  };
  
  // Render pagination controls
  const renderPagination = () => {
    const totalPages = Math.ceil(totalCount / limit);
    
    if (totalPages <= 1) {
      return null;
    }
    
    const pages = [];
    
    // Add first page
    pages.push(
      <button
        key="first"
        onClick={() => handlePageChange(1)}
        disabled={currentPage === 1}
        style={{ fontWeight: currentPage === 1 ? 'bold' : 'normal' }}
      >
        1
      </button>
    );
    
    // Add ellipsis if needed
    if (currentPage > 3) {
      pages.push(<span key="ellipsis1">...</span>);
    }
    
    // Add pages around current page
    for (let i = Math.max(2, currentPage - 1); i <= Math.min(totalPages - 1, currentPage + 1); i++) {
      if (i === 1 || i === totalPages) continue; // Skip first and last pages as they're added separately
      
      pages.push(
        <button
          key={i}
          onClick={() => handlePageChange(i)}
          disabled={currentPage === i}
          style={{ fontWeight: currentPage === i ? 'bold' : 'normal' }}
        >
          {i}
        </button>
      );
    }
    
    // Add ellipsis if needed
    if (currentPage < totalPages - 2) {
      pages.push(<span key="ellipsis2">...</span>);
    }
    
    // Add last page if there are more than one page
    if (totalPages > 1) {
      pages.push(
        <button
          key="last"
          onClick={() => handlePageChange(totalPages)}
          disabled={currentPage === totalPages}
          style={{ fontWeight: currentPage === totalPages ? 'bold' : 'normal' }}
        >
          {totalPages}
        </button>
      );
    }
    
    return (
      <div className="pagination" style={{ display: 'flex', justifyContent: 'center', gap: 'var(--spacing-sm)', marginTop: 'var(--spacing-md)' }}>
        <button
          onClick={() => handlePageChange(currentPage - 1)}
          disabled={currentPage === 1}
        >
          Previous
        </button>
        
        {pages}
        
        <button
          onClick={() => handlePageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
        >
          Next
        </button>
      </div>
    );
  };
  
  // Render star rating
  const renderStarRating = (rating) => {
    const stars = [];
    
    for (let i = 1; i <= 5; i++) {
      stars.push(
        <span
          key={i}
          className={`rating-star ${i <= rating ? 'selected' : ''}`}
          style={{ cursor: 'default' }}
        >
          ★
        </span>
      );
    }
    
    return <div className="feedback-item-rating">{stars}</div>;
  };
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };
  
  // Get category name by ID
  const getCategoryName = (categoryId) => {
    const category = categories.find(cat => cat.id === categoryId);
    return category ? category.name : 'Unknown';
  };
  
  return (
    <div className="feedback-section">
      <h3>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
          <path d="M0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4.414a1 1 0 0 0-.707.293L.854 15.146A.5.5 0 0 1 0 14.793V2zm5 4a1 1 0 1 0-2 0 1 1 0 0 0 2 0zm4 0a1 1 0 1 0-2 0 1 1 0 0 0 2 0zm3 1a1 1 0 1 0 0-2 1 1 0 0 0 0 2z"/>
        </svg>
        Feedback List
      </h3>
      <p className="section-description">
        Browse and filter feedback submitted by users.
      </p>
      
      {error && (
        <div className="error-message" style={{ color: 'var(--error-color)', marginBottom: 'var(--spacing-md)' }}>
          {error}
        </div>
      )}
      
      <div className="filters" style={{ marginBottom: 'var(--spacing-lg)', padding: 'var(--spacing-md)', backgroundColor: 'rgba(0, 0, 0, 0.02)', borderRadius: 'var(--border-radius-sm)' }}>
        <h4 style={{ marginBottom: 'var(--spacing-sm)' }}>Filters</h4>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 'var(--spacing-md)' }}>
          <div className="form-group">
            <label htmlFor="category-filter">Category</label>
            <select
              id="category-filter"
              name="category"
              value={categoryId}
              onChange={handleFilterChange}
            >
              <option value="">All Categories</option>
              {categories.map(category => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="min-rating-filter">Min Rating</label>
            <select
              id="min-rating-filter"
              name="minRating"
              value={minRating}
              onChange={handleFilterChange}
            >
              <option value="">Any</option>
              <option value="1">1 Star</option>
              <option value="2">2 Stars</option>
              <option value="3">3 Stars</option>
              <option value="4">4 Stars</option>
              <option value="5">5 Stars</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="max-rating-filter">Max Rating</label>
            <select
              id="max-rating-filter"
              name="maxRating"
              value={maxRating}
              onChange={handleFilterChange}
            >
              <option value="">Any</option>
              <option value="1">1 Star</option>
              <option value="2">2 Stars</option>
              <option value="3">3 Stars</option>
              <option value="4">4 Stars</option>
              <option value="5">5 Stars</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="start-date-filter">Start Date</label>
            <input
              type="date"
              id="start-date-filter"
              name="startDate"
              value={startDate}
              onChange={handleFilterChange}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="end-date-filter">End Date</label>
            <input
              type="date"
              id="end-date-filter"
              name="endDate"
              value={endDate}
              onChange={handleFilterChange}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="tags-filter">Tags (comma separated)</label>
            <input
              type="text"
              id="tags-filter"
              name="tags"
              value={searchTags}
              onChange={handleFilterChange}
              placeholder="e.g. bug, feature, ui"
            />
          </div>
        </div>
        
        <button
          onClick={resetFilters}
          style={{ marginTop: 'var(--spacing-md)' }}
        >
          Reset Filters
        </button>
      </div>
      
      {loading ? (
        <div style={{ textAlign: 'center', padding: 'var(--spacing-xl)' }}>
          Loading feedback...
        </div>
      ) : feedbackItems.length === 0 ? (
        <div style={{ textAlign: 'center', padding: 'var(--spacing-xl)' }}>
          No feedback found matching the current filters.
        </div>
      ) : (
        <div className="feedback-list">
          {feedbackItems.map(item => (
            <div key={item.id} className="feedback-item">
              <div className="feedback-item-header">
                <div className="feedback-item-title">{item.title}</div>
                {item.rating && renderStarRating(item.rating)}
              </div>
              
              <div className="feedback-item-meta">
                <div>Category: {item.category}</div>
                <div>Date: {formatDate(item.created_at)}</div>
                {item.is_anonymous ? (
                  <div>Anonymous</div>
                ) : item.user_id ? (
                  <div>User: {item.user_id}</div>
                ) : null}
              </div>
              
              <div className="feedback-item-description">
                {item.description}
              </div>
              
              {item.tags && item.tags.length > 0 && (
                <div className="feedback-item-tags">
                  {item.tags.map(tag => (
                    <span key={tag} className="feedback-item-tag">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          
          {renderPagination()}
        </div>
      )}
    </div>
  );
};

export default FeedbackList;