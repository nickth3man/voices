/**
 * Main entry point for the React frontend
 * 
 * This file renders the React application into the DOM.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './components/App';

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Get the root element
  const rootElement = document.getElementById('app');
  
  // Create a React root
  const root = ReactDOM.createRoot(rootElement);
  
  // Render the App component
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
  
  console.log('React application initialized');
});