const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const log = require('electron-log');
const Store = require('electron-store');

// Configure logging
log.transports.file.level = 'info';
log.info('Application starting...');

// Initialize configuration store
const store = new Store({
  defaults: {
    windowBounds: { width: 1200, height: 800 },
    pythonPath: process.platform === 'win32' ? 'python' : 'python3',
    processingOptions: {
      useGPU: true,
      batchSize: 4,
      modelVariant: 'htdemucs',
    },
    storagePath: path.join(app.getPath('userData'), 'storage'),
  }
});

// Keep a global reference of the window object to prevent garbage collection
let mainWindow;
// Keep a reference to the Python process
let pythonProcess = null;

function createWindow() {
  const { width, height } = store.get('windowBounds');
  
  // Create the browser window
  mainWindow = new BrowserWindow({
    width,
    height,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    show: false
  });

  // Load the index.html file or the webpack dev server in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../build/index.html'));
  }

  // Show window when ready to prevent flickering
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Save window size when resized
  mainWindow.on('resize', () => {
    const { width, height } = mainWindow.getBounds();
    store.set('windowBounds', { width, height });
  });

  // Handle window close
  mainWindow.on('closed', () => {
    mainWindow = null;
    stopPythonProcess();
  });

  // Start Python backend
  startPythonProcess();
}

// Start the Python backend process
function startPythonProcess() {
  try {
    const pythonPath = store.get('pythonPath');
    const scriptPath = path.join(__dirname, '../python/app.py');
    
    log.info(`Starting Python process: ${pythonPath} ${scriptPath}`);
    
    pythonProcess = spawn(pythonPath, [scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on('data', (data) => {
      log.info(`Python stdout: ${data}`);
      // Parse JSON responses from Python
      try {
        const response = JSON.parse(data);
        if (mainWindow && response.id) {
          mainWindow.webContents.send('python-response', response);
        }
      } catch (error) {
        log.error(`Error parsing Python output: ${error.message}`);
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      log.error(`Python stderr: ${data}`);
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

// Stop the Python backend process
function stopPythonProcess() {
  if (pythonProcess) {
    log.info('Stopping Python process');
    pythonProcess.kill();
    pythonProcess = null;
  }
}

// IPC handler for sending messages to Python
ipcMain.handle('python-request', async (event, request) => {
  if (!pythonProcess) {
    log.error('Python process not running');
    return { error: 'Python process not running' };
  }

  try {
    // Add a timestamp to the request
    const requestWithTimestamp = {
      ...request,
      timestamp: Date.now()
    };
    
    // Send the request to Python
    pythonProcess.stdin.write(JSON.stringify(requestWithTimestamp) + '\n');
    
    // Return a success status (actual response will come via IPC)
    return { success: true, requestId: request.id };
  } catch (error) {
    log.error(`Error sending request to Python: ${error.message}`);
    return { error: error.message };
  }
});

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  createWindow();

  // On macOS it's common to re-create a window when the dock icon is clicked
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed, except on macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Handle app before quit to ensure Python process is stopped
app.on('before-quit', () => {
  stopPythonProcess();
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  log.error(`Uncaught exception: ${error.message}`);
  log.error(error.stack);
});