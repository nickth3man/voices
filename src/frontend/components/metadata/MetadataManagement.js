import React, { useState } from 'react';
import { 
  Box, 
  Tabs, 
  Tab, 
  Typography, 
  Paper,
  Divider,
  Container
} from '@mui/material';
import MetadataEditor from './MetadataEditor';
import MetadataSearch from './MetadataSearch';

/**
 * TabPanel component for tab content
 */
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`metadata-tabpanel-${index}`}
      aria-labelledby={`metadata-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

/**
 * Metadata Management Component
 * 
 * This component provides a user interface for managing metadata,
 * including searching for files and editing metadata.
 */
const MetadataManagement = () => {
  // State for selected tab
  const [tabValue, setTabValue] = useState(0);
  
  // State for selected file
  const [selectedFileId, setSelectedFileId] = useState(null);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Handle file selection from search
  const handleFileSelect = (fileId) => {
    setSelectedFileId(fileId);
    setTabValue(1); // Switch to editor tab
  };
  
  // Handle metadata update
  const handleMetadataUpdated = () => {
    // If we're on the editor tab, we could refresh the data
    // If we're on the search tab, we could refresh the search results
    // For now, we'll just log a message
    console.log('Metadata updated for file ID:', selectedFileId);
  };
  
  return (
    <Container maxWidth="lg">
      <Paper sx={{ mt: 3, mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            aria-label="metadata management tabs"
            variant="fullWidth"
          >
            <Tab label="Search" id="metadata-tab-0" aria-controls="metadata-tabpanel-0" />
            <Tab 
              label="Editor" 
              id="metadata-tab-1" 
              aria-controls="metadata-tabpanel-1"
              disabled={!selectedFileId}
            />
          </Tabs>
        </Box>
        
        <TabPanel value={tabValue} index={0}>
          <MetadataSearch onFileSelect={handleFileSelect} />
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          {selectedFileId ? (
            <MetadataEditor 
              fileId={selectedFileId} 
              onMetadataUpdated={handleMetadataUpdated}
            />
          ) : (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="body1" color="textSecondary">
                Please select a file from the search tab to edit its metadata.
              </Typography>
            </Box>
          )}
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default MetadataManagement;