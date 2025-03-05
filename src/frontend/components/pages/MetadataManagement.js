import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Tabs, 
  Tab, 
  Paper, 
  Alert,
  CircularProgress,
  Divider
} from '@mui/material';
import MetadataEditor from '../metadata/MetadataEditor';
import MetadataSearch from '../metadata/MetadataSearch';
import PythonBridge from '../../controllers/PythonBridge';

/**
 * Metadata Management Page
 * 
 * This component provides a user interface for managing metadata for audio files.
 * It includes tabs for:
 * - Metadata Editor: For viewing and editing metadata for a specific file
 * - Metadata Search: For searching files based on metadata criteria
 */
const MetadataManagement = () => {
  // State for active tab
  const [activeTab, setActiveTab] = useState(0);
  
  // State for selected file
  const [selectedFileId, setSelectedFileId] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState(null);
  
  // State for loading and error
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // State for statistics
  const [statistics, setStatistics] = useState(null);
  const [loadingStats, setLoadingStats] = useState(false);
  
  // Load file details when selectedFileId changes
  useEffect(() => {
    if (selectedFileId) {
      loadFileDetails();
    }
  }, [selectedFileId]);
  
  // Load statistics on component mount
  useEffect(() => {
    loadStatistics();
  }, []);
  
  // Load file details
  const loadFileDetails = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await PythonBridge.invoke('get_file_details', { fileId: selectedFileId });
      
      if (response.success) {
        setSelectedFileName(response.data.filename);
        
        // Switch to editor tab
        setActiveTab(0);
      } else {
        setError(response.error || 'Failed to load file details');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while loading file details');
    } finally {
      setLoading(false);
    }
  };
  
  // Load metadata statistics
  const loadStatistics = async () => {
    setLoadingStats(true);
    
    try {
      const response = await PythonBridge.invoke('get_metadata_statistics', {});
      
      if (response.success) {
        setStatistics(response.data);
      }
    } catch (err) {
      console.error('Error loading statistics:', err);
    } finally {
      setLoadingStats(false);
    }
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Handle file selection from search
  const handleFileSelect = (fileId) => {
    setSelectedFileId(fileId);
  };
  
  // Handle metadata updated
  const handleMetadataUpdated = () => {
    // Reload statistics
    loadStatistics();
  };
  
  // Format statistics for display
  const formatStatistics = () => {
    if (!statistics) return null;
    
    return (
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 3 }}>
        <Paper sx={{ p: 2, flex: '1 1 200px' }}>
          <Typography variant="subtitle2" color="textSecondary">Total Files</Typography>
          <Typography variant="h4">{statistics.total_files}</Typography>
        </Paper>
        
        <Paper sx={{ p: 2, flex: '1 1 200px' }}>
          <Typography variant="subtitle2" color="textSecondary">Files with Metadata</Typography>
          <Typography variant="h4">{statistics.files_with_metadata}</Typography>
        </Paper>
        
        <Paper sx={{ p: 2, flex: '1 1 200px' }}>
          <Typography variant="subtitle2" color="textSecondary">Files with Custom Fields</Typography>
          <Typography variant="h4">{statistics.files_with_custom_fields}</Typography>
        </Paper>
        
        <Paper sx={{ p: 2, flex: '1 1 200px' }}>
          <Typography variant="subtitle2" color="textSecondary">Average Duration</Typography>
          <Typography variant="h4">
            {statistics.average_duration ? `${statistics.average_duration.toFixed(2)}s` : 'N/A'}
          </Typography>
        </Paper>
      </Box>
    );
  };
  
  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Metadata Management
        </Typography>
        
        {/* Statistics */}
        {loadingStats ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
            <CircularProgress />
          </Box>
        ) : (
          formatStatistics()
        )}
        
        {/* Error Message */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        {/* Tabs */}
        <Paper sx={{ mb: 3 }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
            variant="fullWidth"
          >
            <Tab label="Metadata Editor" disabled={!selectedFileId} />
            <Tab label="Search Files" />
          </Tabs>
          
          <Divider />
          
          {/* Selected File Info */}
          {selectedFileId && (
            <Box sx={{ px: 3, py: 1, bgcolor: 'background.default' }}>
              <Typography variant="body2" color="textSecondary">
                Selected File: <strong>{selectedFileName || `ID: ${selectedFileId}`}</strong>
              </Typography>
            </Box>
          )}
          
          {/* Tab Content */}
          <Box sx={{ p: 3 }}>
            {/* Metadata Editor Tab */}
            {activeTab === 0 && (
              selectedFileId ? (
                <MetadataEditor 
                  fileId={selectedFileId} 
                  onMetadataUpdated={handleMetadataUpdated}
                />
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body1" color="textSecondary">
                    Please select a file from the Search tab to edit its metadata.
                  </Typography>
                </Box>
              )
            )}
            
            {/* Search Tab */}
            {activeTab === 1 && (
              <MetadataSearch onFileSelect={handleFileSelect} />
            )}
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default MetadataManagement;