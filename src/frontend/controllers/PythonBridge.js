/**
 * PythonBridge.js
 * 
 * This module provides a client-side interface for communicating with the Python backend.
 * It wraps the IPC API exposed by the preload script and provides a more convenient
 * interface for sending requests and handling events.
 */

/**
 * Class representing the bridge to the Python backend
 */
class PythonBridge {
  /**
   * Create a new PythonBridge instance
   */
  constructor() {
    this.eventListeners = new Map();
    this.eventRemover = null;
    this.connected = false;
    
    // Initialize the event listener
    this._setupEventListener();
  }
  
  /**
   * Set up the event listener for Python events
   * @private
   */
  _setupEventListener() {
    if (this.eventRemover) {
      // Remove any existing listener
      this.eventRemover();
    }
    
    // Register a new listener
    this.eventRemover = window.api.onPythonEvent((data) => {
      const { event, data: eventData } = data;
      
      // Get the listeners for this event
      const listeners = this.eventListeners.get(event) || [];
      
      // Call each listener
      listeners.forEach(listener => {
        try {
          listener(eventData);
        } catch (error) {
          console.error(`Error in event listener for '${event}':`, error);
        }
      });
    });
  }
  
  /**
   * Check if the Python backend is connected
   * @returns {Promise<boolean>} - A promise that resolves with the connection status
   */
  async isConnected() {
    try {
      const status = await window.api.getAppStatus();
      this.connected = status.pythonRunning;
      return this.connected;
    } catch (error) {
      console.error('Error checking Python connection:', error);
      this.connected = false;
      return false;
    }
  }
  
  /**
   * Send a request to the Python backend
   * @param {string} command - The command to execute
   * @param {Object} params - The parameters for the command
   * @returns {Promise<any>} - A promise that resolves with the response
   */
  async sendRequest(command, params = {}) {
    if (!window.api) {
      throw new Error('API not available. Are you running in Electron?');
    }
    
    try {
      // Send the request to Python
      const response = await window.api.sendToPython(command, params);
      return response;
    } catch (error) {
      console.error(`Error sending request to Python (${command}):`, error);
      throw error;
    }
  }
  
  /**
   * Add an event listener for Python events
   * @param {string} event - The event name
   * @param {Function} callback - The callback function
   * @returns {Function} - A function to remove the listener
   */
  addEventListener(event, callback) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    
    const listeners = this.eventListeners.get(event);
    listeners.push(callback);
    
    // Return a function to remove this specific listener
    return () => {
      const index = listeners.indexOf(callback);
      if (index !== -1) {
        listeners.splice(index, 1);
      }
    };
  }
  
  /**
   * Remove an event listener
   * @param {string} event - The event name
   * @param {Function} callback - The callback function
   */
  removeEventListener(event, callback) {
    if (!this.eventListeners.has(event)) {
      return;
    }
    
    const listeners = this.eventListeners.get(event);
    const index = listeners.indexOf(callback);
    
    if (index !== -1) {
      listeners.splice(index, 1);
    }
  }
  
  /**
   * Remove all event listeners for a specific event
   * @param {string} event - The event name
   */
  removeAllEventListeners(event) {
    if (event) {
      this.eventListeners.delete(event);
    } else {
      this.eventListeners.clear();
    }
  }
  
  /**
   * Ping the Python backend to check connectivity
   * @returns {Promise<Object>} - A promise that resolves with the ping response
   */
  async ping() {
    const timestamp = Date.now();
    try {
      const response = await this.sendRequest('ping', { timestamp });
      const roundTripTime = Date.now() - timestamp;
      return {
        success: true,
        roundTripTime,
        response
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  /**
   * Get the status of the Python backend
   * @returns {Promise<Object>} - A promise that resolves with the status
   */
  async getStatus() {
    try {
      return await this.sendRequest('get_status');
    } catch (error) {
      console.error('Error getting Python status:', error);
      throw error;
    }
  }
}

// Create a singleton instance
const pythonBridge = new PythonBridge();

// Export the singleton
export default pythonBridge;

// Also export the class for testing or if multiple instances are needed
export { PythonBridge };