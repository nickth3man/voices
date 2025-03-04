/**
 * Main App component for the Voices application
 * 
 * This component serves as the root of the React component tree
 * and provides the basic layout structure for the application.
 */

import React, { useState, useEffect } from 'react';
import pythonBridge from '../controllers/PythonBridge';
import Header from './layout/Header';
import Footer from './layout/Footer';
import Dashboard from './pages/Dashboard';

const App = () => {
  const [connectionStatus, setConnectionStatus] = useState({
    connected: false,
    message: 'Initializing...'
  });

  useEffect(() => {
    // Check connection on component mount
    checkConnection();

    // Set up event listener for Python events
    const removeEventListener = pythonBridge.addEventListener('ready', () => {
      setConnectionStatus({
        connected: true,
        message: 'Python backend ready'
      });
    });

    // Clean up event listener on component unmount
    return () => {
      removeEventListener();
    };
  }, []);

  const checkConnection = async () => {
    try {
      // Check if the Python backend is connected
      const isConnected = await pythonBridge.isConnected();
      
      if (isConnected) {
        // Try to ping the Python backend
        const pingResult = await pythonBridge.sendRequest('ping', { timestamp: Date.now() });
        
        if (pingResult && pingResult.pong) {
          setConnectionStatus({
            connected: true,
            message: 'Connected to Python backend'
          });
          
          // Get more detailed status
          try {
            const pythonStatus = await pythonBridge.sendRequest('get_status');
            console.log('Python backend status:', pythonStatus);
          } catch (error) {
            console.warn('Could not get detailed status:', error);
          }
        } else {
          setConnectionStatus({
            connected: false,
            message: 'Python backend not responding'
          });
        }
      } else {
        setConnectionStatus({
          connected: false,
          message: 'Python process not running'
        });
      }
    } catch (error) {
      console.error('Connection check failed:', error);
      setConnectionStatus({
        connected: false,
        message: `Connection error: ${error.message}`
      });
    }
  };

  const handleTestConnection = () => {
    checkConnection();
  };

  return (
    <div className="app-container">
      <Header 
        connected={connectionStatus.connected} 
        statusMessage={connectionStatus.message} 
      />
      <main className="app-content">
        <Dashboard 
          onTestConnection={handleTestConnection} 
          connectionStatus={connectionStatus}
        />
      </main>
      <Footer />
    </div>
  );
};

export default App;