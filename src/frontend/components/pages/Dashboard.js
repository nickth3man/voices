/**
 * Dashboard component for the Voices application
 * 
 * This component serves as the main dashboard/home page of the application.
 * It displays welcome information and connection status.
 */

import React from 'react';

const Dashboard = ({ onTestConnection, connectionStatus }) => {
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
        </ul>
      </div>
    </div>
  );
};

export default Dashboard;