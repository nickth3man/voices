/**
 * Header component for the Voices application
 * 
 * This component displays the application header with the title and connection status.
 */

import React from 'react';

const Header = ({ connected, statusMessage }) => {
  return (
    <header className="app-header">
      <h1>Voices</h1>
      <div className="status-indicator">
        <span 
          className={`status-dot ${connected ? 'connected' : 'disconnected'}`}
        ></span>
        <span id="status-text">{statusMessage}</span>
      </div>
    </header>
  );
};

export default Header;