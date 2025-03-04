/**
 * Integration Tester Component
 * 
 * This component provides a UI for testing the integration between
 * frontend and backend components of the Voices application.
 */

import React, { useState, useEffect } from 'react';
import pythonBridge from '../../controllers/PythonBridge';

const IntegrationTester = () => {
  // State for test results
  const [testResults, setTestResults] = useState({
    connectionStatus: 'Not tested',
    modelRegistryStatus: 'Not tested',
    pipelineStatus: 'Not tested',
    endToEndStatus: 'Not tested'
  });
  
  // State for test progress
  const [testProgress, setTestProgress] = useState({
    running: false,
    currentTest: null,
    log: []
  });
  
  // State for test configuration
  const [testConfig, setTestConfig] = useState({
    testAudioPath: '',
    outputDir: './test_output',
    numSpeakers: 2
  });
  
  // Add a log entry
  const addLogEntry = (message, type = 'info') => {
    setTestProgress(prev => ({
      ...prev,
      log: [...prev.log, { message, type, timestamp: new Date() }]
    }));
  };
  
  // Test connection to Python backend
  const testConnection = async () => {
    setTestResults(prev => ({ ...prev, connectionStatus: 'Testing...' }));
    addLogEntry('Testing connection to Python backend...');
    
    try {
      const pingResult = await pythonBridge.ping();
      
      if (pingResult.success) {
        setTestResults(prev => ({ 
          ...prev, 
          connectionStatus: `Connected (${pingResult.roundTripTime}ms)` 
        }));
        addLogEntry(`Connection successful! Round-trip time: ${pingResult.roundTripTime}ms`, 'success');
        return true;
      } else {
        setTestResults(prev => ({ 
          ...prev, 
          connectionStatus: `Failed: ${pingResult.error}` 
        }));
        addLogEntry(`Connection failed: ${pingResult.error}`, 'error');
        return false;
      }
    } catch (error) {
      setTestResults(prev => ({ 
        ...prev, 
        connectionStatus: `Error: ${error.message}` 
      }));
      addLogEntry(`Connection error: ${error.message}`, 'error');
      return false;
    }
  };
  
  // Test model registry
  const testModelRegistry = async () => {
    setTestResults(prev => ({ ...prev, modelRegistryStatus: 'Testing...' }));
    addLogEntry('Testing model registry...');
    
    try {
      const result = await pythonBridge.sendRequest('test_model_registry', {});
      
      if (result.success) {
        setTestResults(prev => ({ 
          ...prev, 
          modelRegistryStatus: 'Success' 
        }));
        addLogEntry(`Model registry test successful: ${result.message}`, 'success');
        return true;
      } else {
        setTestResults(prev => ({ 
          ...prev, 
          modelRegistryStatus: `Failed: ${result.error}` 
        }));
        addLogEntry(`Model registry test failed: ${result.error}`, 'error');
        return false;
      }
    } catch (error) {
      setTestResults(prev => ({ 
        ...prev, 
        modelRegistryStatus: `Error: ${error.message}` 
      }));
      addLogEntry(`Model registry test error: ${error.message}`, 'error');
      return false;
    }
  };
  
  // Test audio processing pipeline
  const testPipeline = async () => {
    setTestResults(prev => ({ ...prev, pipelineStatus: 'Testing...' }));
    addLogEntry('Testing audio processing pipeline...');
    
    try {
      const result = await pythonBridge.sendRequest('test_pipeline', {
        audio_path: testConfig.testAudioPath,
        output_dir: testConfig.outputDir,
        num_speakers: testConfig.numSpeakers
      });
      
      if (result.success) {
        setTestResults(prev => ({ 
          ...prev, 
          pipelineStatus: 'Success' 
        }));
        addLogEntry(`Pipeline test successful: ${result.message}`, 'success');
        return true;
      } else {
        setTestResults(prev => ({ 
          ...prev, 
          pipelineStatus: `Failed: ${result.error}` 
        }));
        addLogEntry(`Pipeline test failed: ${result.error}`, 'error');
        return false;
      }
    } catch (error) {
      setTestResults(prev => ({ 
        ...prev, 
        pipelineStatus: `Error: ${error.message}` 
      }));
      addLogEntry(`Pipeline test error: ${error.message}`, 'error');
      return false;
    }
  };
  
  // Test end-to-end integration
  const testEndToEnd = async () => {
    setTestResults(prev => ({ ...prev, endToEndStatus: 'Testing...' }));
    addLogEntry('Testing end-to-end integration...');
    
    try {
      const result = await pythonBridge.sendRequest('test_end_to_end', {
        audio_path: testConfig.testAudioPath,
        output_dir: testConfig.outputDir,
        num_speakers: testConfig.numSpeakers
      });
      
      if (result.success) {
        setTestResults(prev => ({ 
          ...prev, 
          endToEndStatus: 'Success' 
        }));
        addLogEntry(`End-to-end test successful: ${result.message}`, 'success');
        return true;
      } else {
        setTestResults(prev => ({ 
          ...prev, 
          endToEndStatus: `Failed: ${result.error}` 
        }));
        addLogEntry(`End-to-end test failed: ${result.error}`, 'error');
        return false;
      }
    } catch (error) {
      setTestResults(prev => ({ 
        ...prev, 
        endToEndStatus: `Error: ${error.message}` 
      }));
      addLogEntry(`End-to-end test error: ${error.message}`, 'error');
      return false;
    }
  };
  
  // Run all tests
  const runAllTests = async () => {
    setTestProgress(prev => ({ ...prev, running: true }));
    addLogEntry('Starting integration tests...', 'info');
    
    // Test connection
    setTestProgress(prev => ({ ...prev, currentTest: 'connection' }));
    const connectionSuccess = await testConnection();
    
    if (!connectionSuccess) {
      addLogEntry('Aborting tests due to connection failure', 'error');
      setTestProgress(prev => ({ ...prev, running: false, currentTest: null }));
      return;
    }
    
    // Test model registry
    setTestProgress(prev => ({ ...prev, currentTest: 'modelRegistry' }));
    await testModelRegistry();
    
    // Test pipeline
    setTestProgress(prev => ({ ...prev, currentTest: 'pipeline' }));
    await testPipeline();
    
    // Test end-to-end
    setTestProgress(prev => ({ ...prev, currentTest: 'endToEnd' }));
    await testEndToEnd();
    
    // Complete
    setTestProgress(prev => ({ ...prev, running: false, currentTest: null }));
    addLogEntry('All tests completed', 'info');
  };
  
  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Get the file path (this only works in Electron)
      const filePath = file.path;
      setTestConfig(prev => ({ ...prev, testAudioPath: filePath }));
      addLogEntry(`Selected test audio file: ${filePath}`, 'info');
    }
  };
  
  // Render log entries
  const renderLog = () => {
    return testProgress.log.map((entry, index) => (
      <div key={index} className={`log-entry log-${entry.type}`}>
        <span className="log-timestamp">
          {entry.timestamp.toLocaleTimeString()}
        </span>
        <span className="log-message">{entry.message}</span>
      </div>
    ));
  };
  
  return (
    <div className="integration-tester">
      <h2>Integration Testing</h2>
      
      <div className="test-config">
        <h3>Test Configuration</h3>
        <div className="form-group">
          <label>Test Audio File:</label>
          <input 
            type="file" 
            accept=".wav,.mp3,.flac" 
            onChange={handleFileSelect}
            disabled={testProgress.running}
          />
          {testConfig.testAudioPath && (
            <div className="file-path">{testConfig.testAudioPath}</div>
          )}
        </div>
        
        <div className="form-group">
          <label>Output Directory:</label>
          <input 
            type="text" 
            value={testConfig.outputDir}
            onChange={(e) => setTestConfig(prev => ({ 
              ...prev, 
              outputDir: e.target.value 
            }))}
            disabled={testProgress.running}
          />
        </div>
        
        <div className="form-group">
          <label>Number of Speakers:</label>
          <input 
            type="number" 
            min="1"
            max="5"
            value={testConfig.numSpeakers}
            onChange={(e) => setTestConfig(prev => ({ 
              ...prev, 
              numSpeakers: parseInt(e.target.value) 
            }))}
            disabled={testProgress.running}
          />
        </div>
      </div>
      
      <div className="test-controls">
        <button 
          onClick={runAllTests} 
          disabled={testProgress.running || !testConfig.testAudioPath}
        >
          Run All Tests
        </button>
        <button 
          onClick={testConnection} 
          disabled={testProgress.running}
        >
          Test Connection Only
        </button>
      </div>
      
      <div className="test-results">
        <h3>Test Results</h3>
        <table>
          <thead>
            <tr>
              <th>Test</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Connection</td>
              <td className={testResults.connectionStatus === 'Success' ? 'success' : 
                            testResults.connectionStatus === 'Not tested' ? '' : 'error'}>
                {testResults.connectionStatus}
              </td>
            </tr>
            <tr>
              <td>Model Registry</td>
              <td className={testResults.modelRegistryStatus === 'Success' ? 'success' : 
                            testResults.modelRegistryStatus === 'Not tested' ? '' : 'error'}>
                {testResults.modelRegistryStatus}
              </td>
            </tr>
            <tr>
              <td>Pipeline</td>
              <td className={testResults.pipelineStatus === 'Success' ? 'success' : 
                            testResults.pipelineStatus === 'Not tested' ? '' : 'error'}>
                {testResults.pipelineStatus}
              </td>
            </tr>
            <tr>
              <td>End-to-End</td>
              <td className={testResults.endToEndStatus === 'Success' ? 'success' : 
                            testResults.endToEndStatus === 'Not tested' ? '' : 'error'}>
                {testResults.endToEndStatus}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      
      <div className="test-log">
        <h3>Test Log</h3>
        <div className="log-container">
          {renderLog()}
        </div>
      </div>
    </div>
  );
};

export default IntegrationTester;