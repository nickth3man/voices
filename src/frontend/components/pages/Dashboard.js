/**
 * Dashboard component for the Voices application
 *
 * This component serves as the main dashboard/home page of the application.
 * It displays welcome information and connection status.
 */

import React from 'react';

const Dashboard = ({ onTestConnection, connectionStatus, onNavigate }) => {
  return (
    <div className="placeholder-content">
      <h2>Welcome to Voices</h2>
      <p>Multi-modal audio processing application for voice isolation and speaker identification.</p>
      
      <div id="connection-status">
        <p>
          {connectionStatus.connected
            ? 'Connected to Python backend'
            : 'Not connected to Python backend'}
        </p>
        <p className="mb-3">{connectionStatus.message}</p>
        <button
          id="test-connection"
          onClick={onTestConnection}
        >
          Test Connection
        </button>
      </div>
      
      <div className="dashboard-features mt-5">
        <h3 className="mb-3">Features</h3>
        <ul className="feature-list">
          <li>Voice Isolation from mixed audio sources</li>
          <li>Speaker Identification across recordings</li>
          <li>Content Organization by speaker, date, or content</li>
          <li>Batch Processing for large files</li>
          <li>
            <a href="#" onClick={(e) => { e.preventDefault(); onNavigate('model-comparison'); }}>
              Model Comparison for voice separation quality
            </a>
          </li>
          <li>
            <a href="#" onClick={(e) => { e.preventDefault(); onNavigate('audio-visualization'); }}>
              Enhanced Audio Visualization with multi-track display
            </a>
          </li>
          <li>
            <a href="#" onClick={(e) => { e.preventDefault(); onNavigate('processing-config'); }}>
              Processing Configuration for model and pipeline settings
            </a>
          </li>
          <li>
            <a href="#" onClick={(e) => { e.preventDefault(); onNavigate('feedback-dashboard'); }}>
              Feedback Dashboard for analytics and user feedback management
            </a>
          </li>
        </ul>
      </div>
      
      <div className="dashboard-actions mt-4">
        <h3 className="mb-3">Quick Actions</h3>
        <div className="action-buttons">
          <button
            className="action-button"
            onClick={() => onNavigate('model-comparison')}
          >
            Compare Models
          </button>
          <button
            className="action-button"
            onClick={() => onNavigate('audio-visualization')}
          >
            Audio Visualization
          </button>
          <button
            className="action-button"
            onClick={() => onNavigate('processing-config')}
          >
            Processing Config
          </button>
          <button
            className="action-button"
            onClick={() => onNavigate('feedback-dashboard')}
          >
            Feedback Dashboard
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;