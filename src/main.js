/**
 * Main entry point for the Voices application
 * 
 * This file sets up the Electron application, creates the main window,
 * initializes the Python backend process, and establishes the IPC bridge
 * for communication between the frontend and backend.
 */

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const log = require('electron-log');
const { v4: uuidv4 } = require('uuid');

// Configure logging
log.transports.file.level = 'info';
log.transports.console.level = 'debug';
log.info('Application starting...');

// Store for pending requests
const pendingRequests = new Map();

// Reference to the Python process
let pythonProcess = null;

// Reference to the main window
let mainWindow = null;

/**
 * Creates the main application window
 */
function createWindow() {
  log.info('Creating main window');
  
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // Always load from public directory for now
  mainWindow.loadFile(path.join(__dirname, '../public/index.html'));
  
  // Open DevTools in development mode
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopPythonProcess();
  });
}

/**
 * Starts the Python backend process
 */
function startPythonProcess() {
  log.info('Starting Python backend process');
  
  // Determine the Python executable path based on the environment
  const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
  
  // Path to the Python backend entry point
  const pythonScriptPath = path.join(__dirname, 'backend/core/communication/server.py');
  
  try {
    // Spawn the Python process
    pythonProcess = spawn(pythonExecutable, [pythonScriptPath]);
    
    // Handle Python process output
    pythonProcess.stdout.on('data', (data) => {
      const message = data.toString().trim();
      log.debug(`Python stdout: ${message}`);
      
      try {
        // Parse the JSON message from Python
        const parsedMessage = JSON.parse(message);
        handlePythonMessage(parsedMessage);
      } catch (error) {
        log.error(`Error parsing Python message: ${error.message}`);
      }
    });
    
    pythonProcess.stderr.on('data', (data) => {
      log.error(`Python stderr: ${data.toString().trim()}`);
    });
    
    pythonProcess.on('close', (code) => {
      log.info(`Python process exited with code ${code}`);
      pythonProcess = null;
    });
    
    pythonProcess.on('error', (error) => {
      log.error(`Failed to start Python process: ${error.message}`);
      pythonProcess = null;
    });
  } catch (error) {
    log.error(`Error starting Python process: ${error.message}`);
  }
}

/**
 * Stops the Python backend process
 */
function stopPythonProcess() {
  if (pythonProcess) {
    log.info('Stopping Python backend process');
    pythonProcess.kill();
    pythonProcess = null;
  }
}

/**
 * Handles messages received from the Python backend
 * @param {Object} message - The parsed message from Python
 */
function handlePythonMessage(message) {
  if (!message || !message.type) {
    log.error('Received invalid message from Python');
    return;
  }
  
  log.debug(`Received message from Python: ${message.type}`);
  
  switch (message.type) {
    case 'response':
      // Handle response to a previous request
      if (message.id && pendingRequests.has(message.id)) {
        const { resolve, reject } = pendingRequests.get(message.id);
        pendingRequests.delete(message.id);
        
        if (message.error) {
          log.error(`Request ${message.id} failed: ${message.error}`);
          reject(new Error(message.error));
        } else {
          log.debug(`Request ${message.id} succeeded`);
          resolve(message.data);
        }
      } else {
        log.warn(`Received response for unknown request ID: ${message.id}`);
      }
      break;
      
    case 'event':
      // Handle asynchronous event from Python
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('python-event', {
          event: message.event,
          data: message.data
        });
      }
      break;
      
    default:
      log.warn(`Unknown message type from Python: ${message.type}`);
  }
}

/**
 * Sends a message to the Python backend
 * @param {string} command - The command to execute
 * @param {Object} params - The parameters for the command
 * @returns {Promise} - A promise that resolves with the response
 */
function sendToPython(command, params = {}) {
  return new Promise((resolve, reject) => {
    if (!pythonProcess) {
      reject(new Error('Python process is not running'));
      return;
    }
    
    // Generate a unique ID for this request
    const requestId = uuidv4();
    
    // Store the promise callbacks
    pendingRequests.set(requestId, { resolve, reject });
    
    // Create the message
    const message = {
      id: requestId,
      command,
      params
    };
    
    // Send the message to Python
    try {
      pythonProcess.stdin.write(JSON.stringify(message) + '\n');
      log.debug(`Sent request ${requestId} to Python: ${command}`);
    } catch (error) {
      pendingRequests.delete(requestId);
      reject(new Error(`Failed to send message to Python: ${error.message}`));
    }
    
    // Set a timeout for the request
    setTimeout(() => {
      if (pendingRequests.has(requestId)) {
        pendingRequests.delete(requestId);
        reject(new Error(`Request ${requestId} timed out`));
      }
    }, 30000); // 30 second timeout
  });
}

// Set up IPC handlers for renderer process
function setupIPC() {
  // Handle requests from the renderer process
  ipcMain.handle('python-request', async (event, { command, params }) => {
    try {
      return await sendToPython(command, params);
    } catch (error) {
      log.error(`Error in python-request: ${error.message}`);
      throw error;
    }
  });
  
  // Handle application status requests
  ipcMain.handle('get-app-status', () => {
    return {
      pythonRunning: pythonProcess !== null,
      version: app.getVersion()
    };
  });
}

// App lifecycle events
app.on('ready', () => {
  log.info('App ready');
  createWindow();
  startPythonProcess();
  setupIPC();
});

app.on('window-all-closed', () => {
  log.info('All windows closed');
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  log.info('App activated');
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('quit', () => {
  log.info('App quitting');
  stopPythonProcess();
});