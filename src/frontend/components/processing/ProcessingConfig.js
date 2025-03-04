/**
 * Processing Configuration Component
 * 
 * This component provides a user interface for configuring processing options,
 * including model selection, processing parameters, output format options,
 * and batch processing settings.
 */

import React, { useState, useEffect } from 'react';
import PythonBridge from '../../controllers/PythonBridge';

const ProcessingConfig = () => {
  // State for available models
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  
  // State for hardware information
  const [hardwareInfo, setHardwareInfo] = useState(null);
  
  // State for optimal settings
  const [optimalSettings, setOptimalSettings] = useState(null);
  const [meetsRequirements, setMeetsRequirements] = useState(true);
  const [requirementsIssues, setRequirementsIssues] = useState([]);
  
  // State for processing parameters
  const [parameters, setParameters] = useState({
    batchSize: 4,
    numWorkers: 2,
    chunkSize: 32000,
    precision: 'mixed'
  });
  
  // State for output format options
  const [outputFormat, setOutputFormat] = useState('wav');
  const [outputQuality, setOutputQuality] = useState('high');
  
  // State for batch processing settings
  const [batchSettings, setBatchSettings] = useState({
    processSubfolders: true,
    createSeparateOutputFolder: true,
    preserveOriginalFiles: true,
    maxConcurrentFiles: 2
  });
  
  // State for loading and error handling
  const [loading, setLoading] = useState({
    models: false,
    hardware: false,
    settings: false,
    saving: false
  });
  const [error, setError] = useState({
    models: null,
    hardware: null,
    settings: null,
    saving: null
  });
  
  // State for configuration presets
  const [presets, setPresets] = useState([
    { name: 'Default', isActive: true },
    { name: 'High Quality', isActive: false },
    { name: 'Fast Processing', isActive: false },
    { name: 'Memory Efficient', isActive: false }
  ]);
  
  // Load available models on component mount
  useEffect(() => {
    loadAvailableModels();
    loadHardwareInfo();
  }, []);
  
  // Load optimal settings when a model is selected
  useEffect(() => {
    if (selectedModel) {
      loadOptimalSettings(selectedModel.id);
    }
  }, [selectedModel]);
  
  // Load available models from backend
  const loadAvailableModels = async () => {
    setLoading(prev => ({ ...prev, models: true }));
    setError(prev => ({ ...prev, models: null }));
    
    try {
      const response = await PythonBridge.sendCommand('get_available_models', {});
      
      if (response.success) {
        setModels(response.models);
        
        // Select the first model by default if available
        if (response.models.length > 0 && !selectedModel) {
          setSelectedModel(response.models[0]);
        }
      } else {
        setError(prev => ({ ...prev, models: response.error }));
      }
    } catch (err) {
      setError(prev => ({ ...prev, models: err.message }));
    } finally {
      setLoading(prev => ({ ...prev, models: false }));
    }
  };
  
  // Load hardware information from backend
  const loadHardwareInfo = async () => {
    setLoading(prev => ({ ...prev, hardware: true }));
    setError(prev => ({ ...prev, hardware: null }));
    
    try {
      const response = await PythonBridge.sendCommand('get_hardware_info', {});
      
      if (response.success) {
        setHardwareInfo(response.hardware);
      } else {
        setError(prev => ({ ...prev, hardware: response.error }));
      }
    } catch (err) {
      setError(prev => ({ ...prev, hardware: err.message }));
    } finally {
      setLoading(prev => ({ ...prev, hardware: false }));
    }
  };
  
  // Load optimal settings for selected model
  const loadOptimalSettings = async (modelId) => {
    setLoading(prev => ({ ...prev, settings: true }));
    setError(prev => ({ ...prev, settings: null }));
    
    try {
      const response = await PythonBridge.sendCommand('get_optimal_settings', {
        model_id: modelId
      });
      
      if (response.success) {
        setOptimalSettings(response.optimal_settings);
        setMeetsRequirements(response.meets_requirements);
        setRequirementsIssues(response.requirements_issues || []);
        
        // Update parameters with optimal settings
        if (response.optimal_settings) {
          setParameters({
            batchSize: response.optimal_settings.batch_size || parameters.batchSize,
            numWorkers: response.optimal_settings.num_workers || parameters.numWorkers,
            chunkSize: response.optimal_settings.chunk_size || parameters.chunkSize,
            precision: response.optimal_settings.precision || parameters.precision
          });
        }
      } else {
        setError(prev => ({ ...prev, settings: response.error }));
      }
    } catch (err) {
      setError(prev => ({ ...prev, settings: err.message }));
    } finally {
      setLoading(prev => ({ ...prev, settings: false }));
    }
  };
  
  // Save configuration to file
  const saveConfiguration = async () => {
    setLoading(prev => ({ ...prev, saving: true }));
    setError(prev => ({ ...prev, saving: null }));
    
    try {
      const configData = {
        model: selectedModel ? selectedModel.id : null,
        parameters: parameters,
        outputFormat: outputFormat,
        outputQuality: outputQuality,
        batchSettings: batchSettings
      };
      
      const response = await PythonBridge.sendCommand('save_configuration', {
        name: 'user_config',
        config: configData
      });
      
      if (!response.success) {
        setError(prev => ({ ...prev, saving: response.error }));
      }
    } catch (err) {
      setError(prev => ({ ...prev, saving: err.message }));
    } finally {
      setLoading(prev => ({ ...prev, saving: false }));
    }
  };
  
  // Handle model selection
  const handleModelSelect = (model) => {
    setSelectedModel(model);
  };
  
  // Handle parameter change
  const handleParameterChange = (name, value) => {
    setParameters(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Handle output format change
  const handleOutputFormatChange = (format) => {
    setOutputFormat(format);
  };
  
  // Handle output quality change
  const handleOutputQualityChange = (event) => {
    setOutputQuality(event.target.value);
  };
  
  // Handle batch setting change
  const handleBatchSettingChange = (name, value) => {
    setBatchSettings(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Handle preset selection
  const handlePresetSelect = (presetName) => {
    // Update active preset
    const updatedPresets = presets.map(preset => ({
      ...preset,
      isActive: preset.name === presetName
    }));
    setPresets(updatedPresets);
    
    // Apply preset settings
    switch (presetName) {
      case 'High Quality':
        setParameters({
          batchSize: 2,
          numWorkers: 4,
          chunkSize: 64000,
          precision: 'single'
        });
        setOutputQuality('high');
        break;
      case 'Fast Processing':
        setParameters({
          batchSize: 8,
          numWorkers: 2,
          chunkSize: 16000,
          precision: 'mixed'
        });
        setOutputQuality('medium');
        break;
      case 'Memory Efficient':
        setParameters({
          batchSize: 1,
          numWorkers: 1,
          chunkSize: 8000,
          precision: 'mixed'
        });
        setOutputQuality('medium');
        break;
      default: // Default preset
        if (optimalSettings) {
          setParameters({
            batchSize: optimalSettings.batch_size || 4,
            numWorkers: optimalSettings.num_workers || 2,
            chunkSize: optimalSettings.chunk_size || 32000,
            precision: optimalSettings.precision || 'mixed'
          });
        } else {
          setParameters({
            batchSize: 4,
            numWorkers: 2,
            chunkSize: 32000,
            precision: 'mixed'
          });
        }
        setOutputQuality('high');
        break;
    }
  };
  
  // Render hardware information
  const renderHardwareInfo = () => {
    if (!hardwareInfo) return null;
    
    return (
      <div className="hardware-info">
        <div className="hardware-card">
          <h4>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
              <path d="M5 0a.5.5 0 0 1 .5.5V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2A2.5 2.5 0 0 1 14 4.5h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14a2.5 2.5 0 0 1-2.5 2.5v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14A2.5 2.5 0 0 1 2 11.5H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2A2.5 2.5 0 0 1 4.5 2V.5A.5.5 0 0 1 5 0zm-.5 3A1.5 1.5 0 0 0 3 4.5v7A1.5 1.5 0 0 0 4.5 13h7a1.5 1.5 0 0 0 1.5-1.5v-7A1.5 1.5 0 0 0 11.5 3h-7zM5 6.5A1.5 1.5 0 0 1 6.5 5h3A1.5 1.5 0 0 1 11 6.5v3A1.5 1.5 0 0 1 9.5 11h-3A1.5 1.5 0 0 1 5 9.5v-3z"/>
            </svg>
            CPU
          </h4>
          <p>Cores: <span className="hardware-value">{hardwareInfo.cpu.physical_cores} physical / {hardwareInfo.cpu.logical_cores} logical</span></p>
          <p>Usage: <span className="hardware-value">{hardwareInfo.cpu.usage_percent}%</span></p>
        </div>
        
        <div className="hardware-card">
          <h4>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
              <path d="M1 3a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h4.586a1 1 0 0 0 .707-.293l.353-.353a.5.5 0 0 1 .708 0l.353.353a1 1 0 0 0 .707.293H15a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1H1Zm1 2h3v5H2V5Zm4 0h4v1H6V5Zm0 2h4v1H6V7Zm0 2h4v1H6V9Zm-1 3V5h9v7H5Z"/>
            </svg>
            Memory
          </h4>
          <p>Total: <span className="hardware-value">{hardwareInfo.memory.total_gb} GB</span></p>
          <p>Available: <span className="hardware-value">{hardwareInfo.memory.available_gb} GB</span></p>
          <p>Used: <span className="hardware-value">{hardwareInfo.memory.percent_used}%</span></p>
        </div>
        
        {hardwareInfo.gpu && hardwareInfo.gpu.length > 0 && (
          <div className="hardware-card">
            <h4>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                <path d="M14.5 2h-13A1.5 1.5 0 0 0 0 3.5v9A1.5 1.5 0 0 0 1.5 14h13a1.5 1.5 0 0 0 1.5-1.5v-9A1.5 1.5 0 0 0 14.5 2zm-13 1h13a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5v-9a.5.5 0 0 1 .5-.5z"/>
                <path d="M7 5.5h2a.5.5 0 0 1 .5.5v1.5a.5.5 0 0 1-.5.5H7a.5.5 0 0 1-.5-.5V6a.5.5 0 0 1 .5-.5zm3 0h2a.5.5 0 0 1 .5.5v1.5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5V6a.5.5 0 0 1 .5-.5zm-6 0h2a.5.5 0 0 1 .5.5v1.5a.5.5 0 0 1-.5.5H4a.5.5 0 0 1-.5-.5V6a.5.5 0 0 1 .5-.5zm3 4.5h2a.5.5 0 0 1 .5.5v1.5a.5.5 0 0 1-.5.5H7a.5.5 0 0 1-.5-.5v-1.5a.5.5 0 0 1 .5-.5zm3 0h2a.5.5 0 0 1 .5.5v1.5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1-.5-.5v-1.5a.5.5 0 0 1 .5-.5zm-6 0h2a.5.5 0 0 1 .5.5v1.5a.5.5 0 0 1-.5.5H4a.5.5 0 0 1-.5-.5v-1.5a.5.5 0 0 1 .5-.5z"/>
              </svg>
              GPU
            </h4>
            <p>Model: <span className="hardware-value">{hardwareInfo.gpu[0].name}</span></p>
            <p>Memory: <span className="hardware-value">{hardwareInfo.gpu[0].memory_total_gb} GB</span></p>
            <p>Available: <span className="hardware-value">{(hardwareInfo.gpu[0].memory_total_gb - hardwareInfo.gpu[0].memory_allocated_gb).toFixed(2)} GB</span></p>
          </div>
        )}
      </div>
    );
  };
  
  // Render model selection cards
  const renderModelSelection = () => {
    if (loading.models) {
      return <p>Loading models...</p>;
    }
    
    if (error.models) {
      return <p className="error">Error loading models: {error.models}</p>;
    }
    
    if (!models || models.length === 0) {
      return <p>No models available.</p>;
    }
    
    return (
      <div className="model-selection">
        {models.map(model => (
          <div 
            key={model.id}
            className={`model-card ${selectedModel && selectedModel.id === model.id ? 'selected' : ''}`}
            onClick={() => handleModelSelect(model)}
          >
            <h4>
              {model.name}
              <span className="model-type">{model.type}</span>
            </h4>
            <p>{model.description}</p>
            
            <div className="model-metrics">
              <div className="model-metric">
                <div className="metric-value">{model.metrics.si_snri.toFixed(2)} dB</div>
                <div className="metric-label">SI-SNRi</div>
              </div>
              <div className="model-metric">
                <div className="metric-value">{model.metrics.sdri.toFixed(2)} dB</div>
                <div className="metric-label">SDRi</div>
              </div>
              <div className="model-metric">
                <div className="metric-value">{model.metrics.processing_speed.toFixed(2)}x</div>
                <div className="metric-label">Speed</div>
              </div>
              <div className="model-metric">
                <div className="metric-value">{model.metrics.memory_usage.toFixed(1)} GB</div>
                <div className="metric-label">Memory</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };
  
  // Render hardware requirements warning
  const renderRequirementsWarning = () => {
    if (meetsRequirements || !requirementsIssues || requirementsIssues.length === 0) {
      return null;
    }
    
    return (
      <div className="requirements-warning" style={{ color: 'var(--warning-color)', marginTop: 'var(--spacing-md)' }}>
        <p><strong>Warning:</strong> Your hardware may not meet the minimum requirements for this model:</p>
        <ul>
          {requirementsIssues.map((issue, index) => (
            <li key={index}>{issue}</li>
          ))}
        </ul>
      </div>
    );
  };
  
  // Render processing parameters
  const renderProcessingParameters = () => {
    return (
      <div className="parameter-grid">
        <div className="parameter-slider">
          <label>
            Batch Size
            <span className="parameter-value">{parameters.batchSize}</span>
          </label>
          <input
            type="range"
            min="1"
            max="16"
            value={parameters.batchSize}
            onChange={(e) => handleParameterChange('batchSize', parseInt(e.target.value))}
          />
          <p className="parameter-description" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-light)' }}>
            Number of audio chunks processed at once. Higher values use more memory but can be faster.
          </p>
        </div>
        
        <div className="parameter-slider">
          <label>
            Workers
            <span className="parameter-value">{parameters.numWorkers}</span>
          </label>
          <input
            type="range"
            min="1"
            max="8"
            value={parameters.numWorkers}
            onChange={(e) => handleParameterChange('numWorkers', parseInt(e.target.value))}
          />
          <p className="parameter-description" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-light)' }}>
            Number of parallel workers. Higher values use more CPU cores.
          </p>
        </div>
        
        <div className="parameter-slider">
          <label>
            Chunk Size
            <span className="parameter-value">{parameters.chunkSize}</span>
          </label>
          <input
            type="range"
            min="4000"
            max="64000"
            step="4000"
            value={parameters.chunkSize}
            onChange={(e) => handleParameterChange('chunkSize', parseInt(e.target.value))}
          />
          <p className="parameter-description" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-light)' }}>
            Size of audio chunks in samples. Larger chunks use more memory but may improve quality.
          </p>
        </div>
        
        <div className="form-group">
          <label>Precision</label>
          <select
            value={parameters.precision}
            onChange={(e) => handleParameterChange('precision', e.target.value)}
          >
            <option value="single">Single (32-bit)</option>
            <option value="mixed">Mixed (16/32-bit)</option>
            <option value="half">Half (16-bit)</option>
          </select>
          <p className="parameter-description" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-light)' }}>
            Numerical precision. Lower precision uses less memory but may reduce quality.
          </p>
        </div>
      </div>
    );
  };
  
  // Render output format options
  const renderOutputFormatOptions = () => {
    const formats = [
      { id: 'wav', name: 'WAV', description: 'Lossless, high quality' },
      { id: 'flac', name: 'FLAC', description: 'Lossless, compressed' },
      { id: 'mp3', name: 'MP3', description: 'Lossy, small size' },
      { id: 'ogg', name: 'OGG', description: 'Lossy, good quality' }
    ];
    
    return (
      <div>
        <div className="format-options">
          {formats.map(format => (
            <div
              key={format.id}
              className={`format-option ${outputFormat === format.id ? 'selected' : ''}`}
              onClick={() => handleOutputFormatChange(format.id)}
            >
              <div className="format-icon">{format.id.toUpperCase()}</div>
              <div className="format-name">{format.name}</div>
              <div className="format-description">{format.description}</div>
            </div>
          ))}
        </div>
        
        <div className="form-group" style={{ marginTop: 'var(--spacing-md)' }}>
          <label>Quality</label>
          <select value={outputQuality} onChange={handleOutputQualityChange}>
            <option value="low">Low (faster, smaller files)</option>
            <option value="medium">Medium (balanced)</option>
            <option value="high">High (slower, larger files)</option>
          </select>
        </div>
      </div>
    );
  };
  
  // Render batch processing settings
  const renderBatchProcessingSettings = () => {
    return (
      <div className="batch-settings">
        <div>
          <div className="checkbox-group">
            <input
              type="checkbox"
              id="processSubfolders"
              checked={batchSettings.processSubfolders}
              onChange={(e) => handleBatchSettingChange('processSubfolders', e.target.checked)}
            />
            <label htmlFor="processSubfolders">Process subfolders</label>
          </div>
          
          <div className="checkbox-group">
            <input
              type="checkbox"
              id="createSeparateOutputFolder"
              checked={batchSettings.createSeparateOutputFolder}
              onChange={(e) => handleBatchSettingChange('createSeparateOutputFolder', e.target.checked)}
            />
            <label htmlFor="createSeparateOutputFolder">Create separate output folder</label>
          </div>
        </div>
        
        <div>
          <div className="checkbox-group">
            <input
              type="checkbox"
              id="preserveOriginalFiles"
              checked={batchSettings.preserveOriginalFiles}
              onChange={(e) => handleBatchSettingChange('preserveOriginalFiles', e.target.checked)}
            />
            <label htmlFor="preserveOriginalFiles">Preserve original files</label>
          </div>
          
          <div className="form-group">
            <label htmlFor="maxConcurrentFiles">Max concurrent files</label>
            <input
              type="number"
              id="maxConcurrentFiles"
              min="1"
              max="10"
              value={batchSettings.maxConcurrentFiles}
              onChange={(e) => handleBatchSettingChange('maxConcurrentFiles', parseInt(e.target.value))}
            />
          </div>
        </div>
      </div>
    );
  };
  
  // Render configuration presets
  const renderConfigurationPresets = () => {
    return (
      <div className="preset-buttons">
        {presets.map(preset => (
          <button
            key={preset.name}
            className={`preset-button ${preset.isActive ? 'active' : ''}`}
            onClick={() => handlePresetSelect(preset.name)}
          >
            {preset.name}
          </button>
        ))}
      </div>
    );
  };
  
  return (
    <div className="processing-config-container">
      <h2>Processing Configuration</h2>
      <p className="description">Configure voice separation processing options, model selection, and output settings.</p>
      
      <div className="config-section">
        <h3>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M9.405 1.05c-.413-1.4-2.397-1.4-2.81 0l-.1.34a1.464 1.464 0 0 1-2.105.872l-.31-.17c-1.283-.698-2.686.705-1.987 1.987l.169.311c.446.82.023 1.841-.872 2.105l-.34.1c-1.4.413-1.4 2.397 0 2.81l.34.1a1.464 1.464 0 0 1 .872 2.105l-.17.31c-.698 1.283.705 2.686 1.987 1.987l.311-.169a1.464 1.464 0 0 1 2.105.872l.1.34c.413 1.4 2.397 1.4 2.81 0l.1-.34a1.464 1.464 0 0 1 2.105-.872l.31.17c1.283.698 2.686-.705 1.987-1.987l-.169-.311a1.464 1.464 0 0 1 .872-2.105l.34-.1c1.4-.413 1.4-2.397 0-2.81l-.34-.1a1.464 1.464 0 0 1-.872-2.105l.17-.31c.698-1.283-.705-2.686-1.987-1.987l-.311.169a1.464 1.464 0 0 1-2.105-.872l-.1-.34zM8 10.93a2.929 2.929 0 1 1 0-5.86 2.929 2.929 0 0 1 0 5.858z"/>
          </svg>
          Configuration Presets
        </h3>
        <p className="section-description">Select a preset configuration or customize settings below.</p>
        
        {renderConfigurationPresets()}
      </div>
      
      <div className="config-section">
        <h3>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M14 1a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h12zM2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2H2z"/>
            <path d="M6.854 4.646a.5.5 0 0 1 0 .708L4.207 8l2.647 2.646a.5.5 0 0 1-.708.708l-3-3a.5.5 0 0 1 0-.708l3-3a.5.5 0 0 1 .708 0zm2.292 0a.5.5 0 0 0 0 .708L11.793 8l-2.647 2.646a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 0 0-.708 0z"/>
          </svg>
          Model Selection
        </h3>
        <p className="section-description">Choose a voice separation model based on your needs.</p>
        
        {renderModelSelection()}
        {renderRequirementsWarning()}
      </div>
      
      <div className="config-section">
        <h3>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M6 12.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 0 1h-3a.5.5 0 0 1-.5-.5ZM3 8.062C3 6.76 4.235 5.765 5.53 5.886a26.58 26.58 0 0 0 4.94 0C11.765 5.765 13 6.76 13 8.062v1.157a.933.933 0 0 1-.765.935c-.845.147-2.34.346-4.235.346-1.895 0-3.39-.2-4.235-.346A.933.933 0 0 1 3 9.219V8.062Zm4.542-.827a.25.25 0 0 0-.217.068l-.92.9a24.767 24.767 0 0 1-1.871-.183.25.25 0 0 0-.068.495c.55.076 1.232.149 2.02.193a.25.25 0 0 0 .189-.071l.754-.736.847 1.71a.25.25 0 0 0 .404.062l.932-.97a25.286 25.286 0 0 0 1.922-.188.25.25 0 0 0-.068-.495c-.538.074-1.207.145-1.98.189a.25.25 0 0 0-.166.076l-.754.785-.842-1.7a.25.25 0 0 0-.182-.135Z"/>
            <path d="M8.5 1.866a1 1 0 1 0-1 0V3h-2A4.5 4.5 0 0 0 1 7.5V8a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1v1a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1a1 1 0 0 0 1-1V9a1 1 0 0 0-1-1v-.5A4.5 4.5 0 0 0 10.5 3h-2V1.866ZM14 7.5V13a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V7.5A3.5 3.5 0 0 1 5.5 4h5A3.5 3.5 0 0 1 14 7.5Z"/>
          </svg>
          Hardware Information
        </h3>
        <p className="section-description">Your system's hardware capabilities for processing.</p>
        
        {renderHardwareInfo()}
      </div>
      
      <div className="config-section">
        <h3>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z"/>
            <path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115l.094-.319z"/>
          </svg>
          Processing Parameters
        </h3>
        <p className="section-description">Adjust processing parameters for performance and quality.</p>
        
        {renderProcessingParameters()}
      </div>
      
      <div className="config-section">
        <h3>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M4.406 1.342A5.53 5.53 0 0 1 8 0c2.69 0 4.923 2 5.166 4.579C14.758 4.804 16 6.137 16 7.773 16 9.569 14.502 11 12.687 11H10a.5.5 0 0 1 0-1h2.688C13.979 10 15 8.988 15 7.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 2.825 10.328 1 8 1a4.53 4.53 0 0 0-2.941 1.1c-.757.652-1.153 1.438-1.153 2.055v.448l-.445.049C2.064 4.805 1 5.952 1 7.318 1 8.785 2.23 10 3.781 10H6a.5.5 0 0 1 0 1H3.781C1.708 11 0 9.366 0 7.318c0-1.763 1.266-3.223 2.942-3.593.143-.863.698-1.723 1.464-2.383z"/>
            <path d="M7.646 15.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 14.293V5.5a.5.5 0 0 0-1 0v8.793l-2.146-2.147a.5.5 0 0 0-.708.708l3 3z"/>
          </svg>
          Output Format
        </h3>
        <p className="section-description">Select output format and quality settings.</p>
        
        {renderOutputFormatOptions()}
      </div>
      
      <div className="config-section">
        <h3>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M3.5 2a.5.5 0 0 0-.5.5v12a.5.5 0 0 0 .5.5h9a.5.5 0 0 0 .5-.5v-12a.5.5 0 0 0-.5-.5H12a.5.5 0 0 1 0-1h.5A1.5 1.5 0 0 1 14 2.5v12a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 14.5v-12A1.5 1.5 0 0 1 3.5 1H4a.5.5 0 0 1 0 1h-.5Z"/>
            <path d="M10 .5a.5.5 0 0 0-.5-.5h-3a.5.5 0 0 0-.5.5.5.5 0 0 1-.5.5.5.5 0 0 0-.5.5V2a.5.5 0 0 0 .5.5h5A.5.5 0 0 0 11 2v-.5a.5.5 0 0 0-.5-.5.5.5 0 0 1-.5-.5Z"/>
            <path d="M4.085 1H3.5A1.5 1.5 0 0 0 2 2.5v12A1.5 1.5 0 0 0 3.5 16h9a1.5 1.5 0 0 0 1.5-1.5v-12A1.5 1.5 0 0 0 12.5 1h-.585c.055.156.085.325.085.5V2a1.5 1.5 0 0 1-1.5 1.5h-5A1.5 1.5 0 0 1 4 2v-.5c0-.175.03-.344.085-.5ZM10 7a1 1 0 1 1 2 0v5a1 1 0 1 1-2 0V7Zm-6 4a1 1 0 1 1 2 0v1a1 1 0 1 1-2 0v-1Zm4-3a1 1 0 0 1 1 1v3a1 1 0 1 1-2 0V9a1 1 0 0 1 1-1Z"/>
          </svg>
          Batch Processing
        </h3>
        <p className="section-description">Configure settings for processing multiple files.</p>
        
        {renderBatchProcessingSettings()}
      </div>
      
      <div className="action-buttons">
        <button className="cancel-button">Cancel</button>
        <button 
          className="save-button"
          onClick={saveConfiguration}
          disabled={loading.saving}
        >
          {loading.saving ? 'Saving...' : 'Save Configuration'}
        </button>
        <button className="apply-button">Apply Settings</button>
      </div>
      
      {error.saving && (
        <p className="error" style={{ color: 'var(--error-color)', marginTop: 'var(--spacing-md)' }}>
          Error saving configuration: {error.saving}
        </p>
      )}
    </div>
  );
};

export default ProcessingConfig;