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
import ModelComparison from './models/ModelComparison';
import AudioVisualization from './audio/AudioVisualization';
import ProcessingConfig from './processing/ProcessingConfig';
import FeedbackForm from './feedback/FeedbackForm';
import FeedbackList from './feedback/FeedbackList';
import FeedbackDashboard from './feedback/FeedbackDashboard';
import IntegrationTester from './testing/IntegrationTester';

const App = () => {
  const [connectionStatus, setConnectionStatus] = useState({
    connected: false,
    message: 'Initializing...'
  });
  const [currentView, setCurrentView] = useState('dashboard');

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

  const handleNavigate = (view) => {
    setCurrentView(view);
  };

  // Render the current view
  const renderView = () => {
    switch (currentView) {
      case 'model-comparison':
        return <ModelComparison />;
      case 'audio-visualization':
        return <AudioVisualization />;
      case 'processing-config':
        return <ProcessingConfig />;
      case 'feedback-dashboard':
        return <FeedbackDashboard />;
      case 'feedback-form':
        return <FeedbackForm onSubmitSuccess={() => handleNavigate('feedback-dashboard')} />;
      case 'feedback-list':
        return <FeedbackList />;
      case 'integration-tester':
        return <IntegrationTester />;
      case 'dashboard':
      default:
        return (
          <Dashboard
            onTestConnection={handleTestConnection}
            connectionStatus={connectionStatus}
            onNavigate={handleNavigate}
          />
        );
    }
  };

  return (
    <div className="app-container">
      <Header
        connected={connectionStatus.connected}
        statusMessage={connectionStatus.message}
        currentView={currentView}
        onNavigate={handleNavigate}
      />
      <main className="app-content">
        {renderView()}
      </main>
      <Footer />
    </div>
  );
};

export default App;