/**
 * Model Comparison component for the Voices application
 * 
 * This component provides a user interface for comparing the results of different
 * voice separation models on the same audio input. It includes visualizations of
 * separation quality, objective metrics, and A/B testing capabilities.
 */

import React, { useState, useEffect, useRef } from 'react';
import pythonBridge from '../../controllers/PythonBridge';

const ModelComparison = () => {
  // State for available models
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // State for audio file
  const [audioFile, setAudioFile] = useState(null);
  const [audioPath, setAudioPath] = useState('');
  
  // State for comparison results
  const [comparisonResults, setComparisonResults] = useState(null);
  const [metrics, setMetrics] = useState({});
  
  // State for A/B testing
  const [currentPlayingModel, setCurrentPlayingModel] = useState(null);
  const [currentPlayingSource, setCurrentPlayingSource] = useState(null);
  const audioRef = useRef(null);
  
  // Fetch available models on component mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);
  
  // Fetch available models from the backend
  const fetchAvailableModels = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await pythonBridge.sendRequest('get_available_models', {});
      
      if (response.success) {
        setAvailableModels(response.data);
      } else {
        setError(response.error || 'Failed to fetch available models');
      }
    } catch (err) {
      setError(`Error fetching models: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle model selection
  const handleModelSelect = (modelId) => {
    if (selectedModels.includes(modelId)) {
      setSelectedModels(selectedModels.filter(id => id !== modelId));
    } else {
      setSelectedModels([...selectedModels, modelId]);
    }
  };
  
  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
      setAudioPath(file.path);
    }
  };
  
  // Run comparison
  const runComparison = async () => {
    if (!audioPath) {
      setError('Please select an audio file');
      return;
    }
    
    if (selectedModels.length < 1) {
      setError('Please select at least one model');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setComparisonResults(null);
    
    try {
      // Run model comparison
      const response = await pythonBridge.sendRequest('compare_models', {
        audioPath,
        modelIds: selectedModels
      });
      
      if (response.success) {
        setComparisonResults(response.data);
        
        // Fetch metrics for the selected models
        fetchModelMetrics(selectedModels);
      } else {
        setError(response.error || 'Failed to run comparison');
      }
    } catch (err) {
      setError(`Error running comparison: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Fetch metrics for models
  const fetchModelMetrics = async (modelIds) => {
    try {
      const response = await pythonBridge.sendRequest('get_model_metrics', {
        modelIds
      });
      
      if (response.success) {
        setMetrics(response.data);
      } else {
        console.error('Failed to fetch metrics:', response.error);
      }
    } catch (err) {
      console.error('Error fetching metrics:', err);
    }
  };
  
  // Play audio from a specific model and source
  const playAudio = (modelId, sourcePath, sourceIndex) => {
    if (audioRef.current) {
      audioRef.current.pause();
      
      // Set the audio source
      audioRef.current.src = `file://${sourcePath}`;
      
      // Update state
      setCurrentPlayingModel(modelId);
      setCurrentPlayingSource(sourceIndex);
      
      // Play the audio
      audioRef.current.play();
    }
  };
  
  // Play original audio
  const playOriginalAudio = () => {
    if (audioRef.current && audioPath) {
      audioRef.current.pause();
      
      // Set the audio source
      audioRef.current.src = `file://${audioPath}`;
      
      // Update state
      setCurrentPlayingModel('original');
      setCurrentPlayingSource(null);
      
      // Play the audio
      audioRef.current.play();
    }
  };
  
  // Render metrics for a model
  const renderMetrics = (modelId) => {
    const modelMetrics = metrics[modelId];
    
    if (!modelMetrics) {
      return <p>No metrics available</p>;
    }
    
    if (modelMetrics.error) {
      return <p>Error: {modelMetrics.error}</p>;
    }
    
    return (
      <div className="metrics-container">
        <h4>Objective Metrics</h4>
        <table className="metrics-table">
          <tbody>
            {Object.entries(modelMetrics).map(([key, value]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };
  
  // Render comparison results
  const renderComparisonResults = () => {
    if (!comparisonResults) {
      return null;
    }
    
    const { results, originalPath } = comparisonResults;
    
    return (
      <div className="comparison-results">
        <h3>Comparison Results</h3>
        
        <div className="original-audio">
          <h4>Original Audio</h4>
          <button 
            onClick={playOriginalAudio}
            className={currentPlayingModel === 'original' ? 'playing' : ''}
          >
            {currentPlayingModel === 'original' ? 'Playing...' : 'Play'}
          </button>
        </div>
        
        <div className="model-results-grid">
          {Object.entries(results).map(([modelId, modelResult]) => (
            <div key={modelId} className="model-result-card">
              <h4>{modelResult.name || modelId}</h4>
              
              {modelResult.error ? (
                <p className="error">Error: {modelResult.error}</p>
              ) : (
                <>
                  <div className="source-buttons">
                    <h5>Separated Sources</h5>
                    {modelResult.sourcePaths.map((sourcePath, index) => (
                      <button 
                        key={index}
                        onClick={() => playAudio(modelId, sourcePath, index)}
                        className={currentPlayingModel === modelId && currentPlayingSource === index ? 'playing' : ''}
                      >
                        {currentPlayingModel === modelId && currentPlayingSource === index 
                          ? `Playing Source ${index + 1}...` 
                          : `Play Source ${index + 1}`}
                      </button>
                    ))}
                  </div>
                  
                  {renderMetrics(modelId)}
                </>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };
  
  return (
    <div className="model-comparison-container">
      <h2>Model Comparison</h2>
      <p>Compare the results of different voice separation models on the same audio input.</p>
      
      <div className="comparison-setup">
        <div className="audio-selection">
          <h3>Select Audio File</h3>
          <input 
            type="file" 
            accept="audio/*" 
            onChange={handleFileSelect} 
            disabled={isLoading}
          />
          {audioFile && (
            <p>Selected: {audioFile.name}</p>
          )}
        </div>
        
        <div className="model-selection">
          <h3>Select Models to Compare</h3>
          {isLoading && <p>Loading models...</p>}
          {error && <p className="error">{error}</p>}
          
          <div className="model-list">
            {availableModels.map(model => (
              <div key={model.id} className="model-item">
                <label>
                  <input 
                    type="checkbox" 
                    checked={selectedModels.includes(model.id)}
                    onChange={() => handleModelSelect(model.id)}
                    disabled={isLoading}
                  />
                  {model.name} ({model.type})
                </label>
              </div>
            ))}
          </div>
          
          <button 
            onClick={runComparison} 
            disabled={isLoading || !audioPath || selectedModels.length < 1}
            className="run-comparison-btn"
          >
            {isLoading ? 'Running...' : 'Run Comparison'}
          </button>
        </div>
      </div>
      
      {renderComparisonResults()}
      
      {/* Hidden audio element for playback */}
      <audio ref={audioRef} controls style={{ display: 'none' }} />
    </div>
  );
};

export default ModelComparison;