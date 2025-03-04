/**
 * Header component for the Voices application
 *
 * This component displays the application header with the title and connection status.
 */

import React from 'react';

const Header = ({ connected, statusMessage, currentView, onNavigate }) => {
  return (
    <header className="app-header">
      <div className="header-left">
        <h1>Voices</h1>
        <nav className="main-nav">
          <ul>
            <li>
              <button
                className={currentView === 'dashboard' ? 'active' : ''}
                onClick={() => onNavigate('dashboard')}
              >
                Dashboard
              </button>
            </li>
            <li>
              <button
                className={currentView === 'model-comparison' ? 'active' : ''}
                onClick={() => onNavigate('model-comparison')}
              >
                Model Comparison
              </button>
            </li>
          </ul>
        </nav>
      </div>
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