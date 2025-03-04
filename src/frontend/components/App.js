import React, { useState, useEffect } from 'react';
import '../styles/App.css';
import Sidebar from './Sidebar';
import FileBrowser from './FileBrowser';
import ProcessingControls from './ProcessingControls';
import AudioVisualizer from './AudioVisualizer';
import StatusBar from './StatusBar';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('idle');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [systemInfo, setSystemInfo] = useState(null);

  // Fetch system info on component mount
  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const response = await window.api.sendToPython({
          id: 'system-info-' + Date.now(),
          action: 'get_system_info',
          params: {}
        });
        
        if (response.success) {
          console.log('System info received');
        }
      } catch (error) {
        console.error('Error fetching system info:', error);
      }
    };

    // Set up listener for Python responses
    const unsubscribe = window.api.onPythonResponse((response) => {
      console.log('Received response from Python:', response);
      
      // Handle different response types
      if (response.progress !== undefined) {
        setProcessingProgress(response.progress);
        setProcessingStatus(response.status);
      } else if (response.data && response.data.platform) {
        setSystemInfo(response.data);
      }
    });

    fetchSystemInfo();

    // Clean up listener on unmount
    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, []);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  const handleProcessFile = async (options) => {
    if (!selectedFile) return;
    
    setProcessingStatus('starting');
    setProcessingProgress(0);
    
    try {
      const response = await window.api.sendToPython({
        id: 'process-file-' + Date.now(),
        action: 'process_file',
        params: {
          file_path: selectedFile.path,
          options: options
        }
      });
      
      if (response.success) {
        setProcessingStatus('processing');
      } else {
        setProcessingStatus('error');
        console.error('Error starting processing:', response.error);
      }
    } catch (error) {
      setProcessingStatus('error');
      console.error('Error processing file:', error);
    }
  };

  return (
    <div className="app">
      <div className="app-header">
        <h1>Voices</h1>
        <div className="app-controls">
          {/* App controls will go here */}
        </div>
      </div>
      
      <div className="app-content">
        <Sidebar />
        
        <div className="main-content">
          <div className="top-panel">
            <FileBrowser onFileSelect={handleFileSelect} />
            <ProcessingControls 
              onProcess={handleProcessFile}
              disabled={!selectedFile || processingStatus === 'processing'}
              gpuAvailable={systemInfo?.gpu_available}
            />
          </div>
          
          <div className="bottom-panel">
            <AudioVisualizer 
              file={selectedFile}
              processingStatus={processingStatus}
            />
          </div>
        </div>
      </div>
      
      <StatusBar 
        status={processingStatus}
        progress={processingProgress}
        systemInfo={systemInfo}
      />
    </div>
  );
}

export default App;