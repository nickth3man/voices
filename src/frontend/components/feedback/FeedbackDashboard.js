/**
 * Feedback Dashboard Component
 * 
 * This component displays analytics and management tools for user feedback.
 * It includes summary metrics, charts, and navigation to other feedback features.
 */

import React, { useState, useEffect } from 'react';
import pythonBridge from '../../controllers/PythonBridge';

const FeedbackDashboard = () => {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dateRange, setDateRange] = useState({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], // 30 days ago
    endDate: new Date().toISOString().split('T')[0] // today
  });
  const [selectedCategory, setSelectedCategory] = useState(null);

  // Fetch analytics data
  useEffect(() => {
    fetchAnalytics();
  }, [dateRange, selectedCategory]);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      const response = await pythonBridge.sendRequest('feedback', {
        action: 'get_feedback_analytics',
        category_id: selectedCategory,
        start_date: dateRange.startDate,
        end_date: dateRange.endDate
      });

      if (response && response.success) {
        setAnalytics(response.analytics);
        setError(null);
      } else {
        setError(response?.error || 'Failed to fetch analytics');
      }
    } catch (err) {
      console.error('Error fetching feedback analytics:', err);
      setError('An error occurred while fetching analytics data');
    } finally {
      setLoading(false);
    }
  };

  const handleDateRangeChange = (e) => {
    const { name, value } = e.target;
    setDateRange(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleCategoryChange = (e) => {
    setSelectedCategory(e.target.value === 'all' ? null : parseInt(e.target.value));
  };

  // Render loading state
  if (loading && !analytics) {
    return (
      <div className="feedback-container">
        <h2>Feedback Dashboard</h2>
        <div className="loading-indicator">Loading analytics data...</div>
      </div>
    );
  }

  // Render error state
  if (error && !analytics) {
    return (
      <div className="feedback-container">
        <h2>Feedback Dashboard</h2>
        <div className="error-message">
          <p>Error: {error}</p>
          <button onClick={fetchAnalytics} className="submit-button">Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="feedback-container">
      <h2>Feedback Dashboard</h2>
      <p className="description">Analyze user feedback and manage feature requests</p>

      {/* Filters */}
      <div className="feedback-section">
        <h3>Filters</h3>
        <div className="filters-container">
          <div className="form-group">
            <label htmlFor="startDate">Start Date</label>
            <input
              type="date"
              id="startDate"
              name="startDate"
              value={dateRange.startDate}
              onChange={handleDateRangeChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="endDate">End Date</label>
            <input
              type="date"
              id="endDate"
              name="endDate"
              value={dateRange.endDate}
              onChange={handleDateRangeChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="category">Category</label>
            <select
              id="category"
              name="category"
              value={selectedCategory || 'all'}
              onChange={handleCategoryChange}
            >
              <option value="all">All Categories</option>
              {analytics?.categories?.map(category => (
                <option key={category.name} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </div>
          <button onClick={fetchAnalytics} className="submit-button">Apply Filters</button>
        </div>
      </div>

      {/* Summary Metrics */}
      <div className="feedback-dashboard">
        <div className="dashboard-card">
          <div className="dashboard-card-title">Total Feedback</div>
          <div className="dashboard-card-value">{analytics?.total_count || 0}</div>
          <div className="dashboard-card-description">
            Total feedback received in selected period
          </div>
        </div>
        <div className="dashboard-card">
          <div className="dashboard-card-title">Average Rating</div>
          <div className="dashboard-card-value">
            {analytics?.average_rating ? analytics.average_rating.toFixed(1) : 'N/A'}
          </div>
          <div className="dashboard-card-description">
            Average rating across all feedback
          </div>
        </div>
        <div className="dashboard-card">
          <div className="dashboard-card-title">Positive Feedback</div>
          <div className="dashboard-card-value">{analytics?.positive_count || 0}</div>
          <div className="dashboard-card-description">
            Feedback with ratings 4-5
          </div>
        </div>
        <div className="dashboard-card">
          <div className="dashboard-card-title">Neutral Feedback</div>
          <div className="dashboard-card-value">{analytics?.neutral_count || 0}</div>
          <div className="dashboard-card-description">
            Feedback with rating 3
          </div>
        </div>
        <div className="dashboard-card">
          <div className="dashboard-card-title">Negative Feedback</div>
          <div className="dashboard-card-value">{analytics?.negative_count || 0}</div>
          <div className="dashboard-card-description">
            Feedback with ratings 1-2
          </div>
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="chart-container">
        <h3 className="chart-title">Feedback by Category</h3>
        {analytics?.categories && analytics.categories.length > 0 ? (
          <div className="category-breakdown">
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Count</th>
                  <th>Average Rating</th>
                </tr>
              </thead>
              <tbody>
                {analytics.categories.map(category => (
                  <tr key={category.name}>
                    <td>{category.name}</td>
                    <td>{category.count}</td>
                    <td>{category.average_rating ? category.average_rating.toFixed(1) : 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="no-data-message">No category data available for the selected period</p>
        )}
      </div>

      {/* Top Tags */}
      <div className="chart-container">
        <h3 className="chart-title">Top Tags</h3>
        {analytics?.top_tags && analytics.top_tags.length > 0 ? (
          <div className="tags-breakdown">
            <div className="tag-cloud">
              {analytics.top_tags.map(tag => (
                <div 
                  key={tag.name} 
                  className="tag-item"
                  style={{ 
                    fontSize: `${Math.max(1, Math.min(2, 1 + (tag.count / Math.max(...analytics.top_tags.map(t => t.count))) * 1))}em` 
                  }}
                >
                  {tag.name} ({tag.count})
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="no-data-message">No tag data available for the selected period</p>
        )}
      </div>

      {/* Actions */}
      <div className="feedback-section">
        <h3>Actions</h3>
        <div className="action-buttons">
          <button 
            className="action-button"
            onClick={() => window.location.hash = '#/feedback-form'}
          >
            Submit New Feedback
          </button>
          <button 
            className="action-button"
            onClick={() => window.location.hash = '#/feedback-list'}
          >
            View All Feedback
          </button>
          <button 
            className="action-button"
            onClick={() => window.location.hash = '#/feature-requests'}
          >
            Manage Feature Requests
          </button>
        </div>
      </div>

      {/* Feature Requests Summary */}
      <div className="feedback-section">
        <h3>Feature Requests Summary</h3>
        <div className="feature-requests-summary">
          <div className="feature-request-stats">
            <div className="stat-item">
              <div className="stat-label">New</div>
              <div className="stat-value status-new">5</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Under Review</div>
              <div className="stat-value status-under-review">3</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Planned</div>
              <div className="stat-value status-planned">7</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">In Progress</div>
              <div className="stat-value status-in-progress">2</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Completed</div>
              <div className="stat-value status-completed">12</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Declined</div>
              <div className="stat-value status-declined">4</div>
            </div>
          </div>
          <button 
            className="view-all-button"
            onClick={() => window.location.hash = '#/feature-requests'}
          >
            View All Feature Requests
          </button>
        </div>
      </div>

      <style jsx>{`
        .filters-container {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: var(--spacing-md);
          margin-bottom: var(--spacing-md);
        }
        
        .analytics-table {
          width: 100%;
          border-collapse: collapse;
          margin-top: var(--spacing-md);
        }
        
        .analytics-table th, .analytics-table td {
          padding: var(--spacing-sm);
          text-align: left;
          border-bottom: 1px solid var(--border-color);
        }
        
        .analytics-table th {
          font-weight: 600;
          background-color: rgba(0, 0, 0, 0.02);
        }
        
        .tag-cloud {
          display: flex;
          flex-wrap: wrap;
          gap: var(--spacing-sm);
          padding: var(--spacing-md);
        }
        
        .tag-item {
          padding: var(--spacing-xs) var(--spacing-sm);
          background-color: rgba(52, 152, 219, 0.1);
          color: var(--primary-color);
          border-radius: var(--border-radius-sm);
          white-space: nowrap;
        }
        
        .feature-request-stats {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
          gap: var(--spacing-md);
          margin-bottom: var(--spacing-md);
        }
        
        .stat-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: var(--spacing-md);
          border: 1px solid var(--border-color);
          border-radius: var(--border-radius-sm);
          background-color: var(--card-background);
        }
        
        .stat-label {
          font-size: var(--font-size-sm);
          margin-bottom: var(--spacing-xs);
          color: var(--text-light);
        }
        
        .stat-value {
          font-size: var(--font-size-xl);
          font-weight: 700;
          padding: var(--spacing-xs) var(--spacing-sm);
          border-radius: var(--border-radius-sm);
        }
        
        .view-all-button {
          display: block;
          margin: var(--spacing-md) auto;
          padding: var(--spacing-sm) var(--spacing-md);
          background-color: var(--primary-color);
          color: white;
          border: none;
          border-radius: var(--border-radius-sm);
          font-weight: 600;
          cursor: pointer;
        }
        
        .no-data-message {
          padding: var(--spacing-md);
          text-align: center;
          color: var(--text-light);
          font-style: italic;
        }
      `}</style>
    </div>
  );
};

export default FeedbackDashboard;