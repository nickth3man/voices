const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'api', {
    // Send a request to the Python backend
    sendToPython: async (request) => {
      try {
        return await ipcRenderer.invoke('python-request', request);
      } catch (error) {
        console.error('Error sending request to Python:', error);
        return { error: error.message };
      }
    },
    
    // Listen for responses from the Python backend
    onPythonResponse: (callback) => {
      const subscription = (event, response) => callback(response);
      ipcRenderer.on('python-response', subscription);
      
      // Return a function to remove the event listener
      return () => {
        ipcRenderer.removeListener('python-response', subscription);
      };
    },
    
    // Get app version
    getAppVersion: () => {
      return process.env.npm_package_version;
    },
    
    // Get platform information
    getPlatformInfo: () => {
      return {
        platform: process.platform,
        arch: process.arch,
        version: process.version
      };
    }
  }
);

// Log when preload script has run
console.log('Preload script loaded');