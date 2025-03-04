/**
 * Preload script for the Voices application
 * 
 * This file exposes a secure API to the renderer process,
 * allowing it to communicate with the main process and the Python backend
 * while maintaining proper context isolation.
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'api', 
  {
    /**
     * Send a request to the Python backend
     * @param {string} command - The command to execute
     * @param {Object} params - The parameters for the command
     * @returns {Promise} - A promise that resolves with the response
     */
    sendToPython: (command, params = {}) => {
      return ipcRenderer.invoke('python-request', { command, params });
    },
    
    /**
     * Get the current application status
     * @returns {Promise<Object>} - A promise that resolves with the status
     */
    getAppStatus: () => {
      return ipcRenderer.invoke('get-app-status');
    },
    
    /**
     * Register a listener for Python events
     * @param {Function} callback - The callback function
     * @returns {Function} - A function to remove the listener
     */
    onPythonEvent: (callback) => {
      // Wrap the callback to ensure it's safe
      const wrappedCallback = (_, data) => callback(data);
      
      // Add the event listener
      ipcRenderer.on('python-event', wrappedCallback);
      
      // Return a function to remove the listener
      return () => {
        ipcRenderer.removeListener('python-event', wrappedCallback);
      };
    }
  }
);