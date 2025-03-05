/**
 * Renderer script for the Voices application
 *
 * This script handles the initial setup and verification of the Electron environment
 * before the React application is loaded.
 */

/**
 * Initialize the application
 */
function initialize() {
  console.log('Initializing Voices application...');
  
  // Check if running in Electron
  if (!window.api) {
    console.error('API not available. Are you running in Electron?');
    document.body.innerHTML = `
      <div class="error-container">
        <h1>Error: Not Running in Electron</h1>
        <p>This application must be run in an Electron environment.</p>
      </div>
    `;
    return;
  }
  
  console.log('Electron environment detected, proceeding with initialization...');
  
  // Set up global error handler
  window.addEventListener('error', (event) => {
    console.error('Uncaught error:', event.error);
  });
  
  // Set up unhandled promise rejection handler
  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
  });
  
  // The React application will be loaded by the script tag in index.html
  console.log('Basic initialization complete, waiting for React to load...');
}

// Initialize when the DOM is ready
document.addEventListener('DOMContentLoaded', initialize);