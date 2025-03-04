import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CardHeader, 
  Divider, 
  Grid, 
  TextField, 
  Typography, 
  CircularProgress, 
  Alert,
  Paper,
  IconButton,
  Tooltip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip
} from '@mui/material';
import { 
  Save as SaveIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Upload as UploadIcon
} from '@mui/icons-material';
import MetadataBridge from '../../controllers/MetadataBridge';

/**
 * Metadata Editor Component
 * 
 * This component provides a user interface for viewing and editing
 * metadata for a specific audio file.
 */
const MetadataEditor = ({ fileId, onMetadataUpdated }) => {
  // State for metadata
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  
  // State for custom metadata dialog
  const [customMetadataDialog, setCustomMetadataDialog] = useState({
    open: false,
    fieldName: '',
    fieldValue: '',
    fieldType: 'text'
  });
  
  // State for delete confirmation dialog
  const [deleteDialog, setDeleteDialog] = useState({
    open: false,
    fieldName: ''
  });
  
  // Load metadata on component mount or fileId change
  useEffect(() => {
    if (fileId) {
      loadMetadata();
    }
  }, [fileId]);
  
  // Load metadata from the backend
  const loadMetadata = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await MetadataBridge.getMetadata(fileId);
      
      if (response.success) {
        setMetadata(response.data || {});
      } else {
        setError(response.error || 'Failed to load metadata');
        setMetadata(null);
      }
    } catch (err) {
      setError(err.message || 'An error occurred while loading metadata');
      setMetadata(null);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle metadata field change
  const handleMetadataChange = (field, value) => {
    setMetadata(prevMetadata => ({
      ...prevMetadata,
      [field]: value
    }));
  };
  
  // Handle save button click
  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSaveSuccess(false);
    
    try {
      const response = await MetadataBridge.updateMetadata(fileId, metadata);
      
      if (response.success) {
        setSaveSuccess(true);
        if (onMetadataUpdated) {
          onMetadataUpdated();
        }
      } else {
        setError(response.error || 'Failed to save metadata');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while saving metadata');
    } finally {
      setSaving(false);
    }
  };
  
  // Handle refresh button click
  const handleRefresh = () => {
    loadMetadata();
  };
  
  // Handle add custom metadata button click
  const handleAddCustomMetadata = () => {
    setCustomMetadataDialog({
      open: true,
      fieldName: '',
      fieldValue: '',
      fieldType: 'text'
    });
  };
  
  // Handle custom metadata dialog close
  const handleCustomMetadataDialogClose = () => {
    setCustomMetadataDialog({
      ...customMetadataDialog,
      open: false
    });
  };
  
  // Handle custom metadata dialog field change
  const handleCustomMetadataDialogChange = (field, value) => {
    setCustomMetadataDialog(prevDialog => ({
      ...prevDialog,
      [field]: value
    }));
  };
  
  // Handle custom metadata dialog save
  const handleCustomMetadataDialogSave = async () => {
    if (!customMetadataDialog.fieldName) {
      setError('Field name is required');
      return;
    }
    
    setSaving(true);
    setError(null);
    
    try {
      const response = await MetadataBridge.addCustomMetadata(
        fileId,
        customMetadataDialog.fieldName,
        customMetadataDialog.fieldValue,
        customMetadataDialog.fieldType
      );
      
      if (response.success) {
        // Close dialog
        setCustomMetadataDialog({
          ...customMetadataDialog,
          open: false
        });
        
        // Reload metadata
        loadMetadata();
      } else {
        setError(response.error || 'Failed to add custom metadata');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while adding custom metadata');
    } finally {
      setSaving(false);
    }
  };
  
  // Handle delete custom metadata button click
  const handleDeleteCustomMetadata = (fieldName) => {
    setDeleteDialog({
      open: true,
      fieldName: fieldName
    });
  };
  
  // Handle delete dialog close
  const handleDeleteDialogClose = () => {
    setDeleteDialog({
      ...deleteDialog,
      open: false
    });
  };
  
  // Handle delete dialog confirm
  const handleDeleteDialogConfirm = async () => {
    setSaving(true);
    setError(null);
    
    try {
      const response = await MetadataBridge.removeCustomMetadata(
        fileId,
        deleteDialog.fieldName
      );
      
      if (response.success) {
        // Close dialog
        setDeleteDialog({
          ...deleteDialog,
          open: false
        });
        
        // Reload metadata
        loadMetadata();
      } else {
        setError(response.error || 'Failed to remove custom metadata');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while removing custom metadata');
    } finally {
      setSaving(false);
    }
  };
  
  // Handle export metadata button click
  const handleExportMetadata = async () => {
    setSaving(true);
    setError(null);
    
    try {
      const response = await MetadataBridge.exportMetadata(fileId);
      
      if (response.success) {
        // Show success message
        setSaveSuccess(true);
      } else {
        setError(response.error || 'Failed to export metadata');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while exporting metadata');
    } finally {
      setSaving(false);
    }
  };
  
  // Get custom metadata fields
  const getCustomMetadataFields = () => {
    if (!metadata) return [];
    
    return Object.keys(metadata)
      .filter(key => key.startsWith('custom_'))
      .map(key => ({
        name: key.replace('custom_', ''),
        value: metadata[key],
        type: metadata[`custom_${key.replace('custom_', '')}_type`] || 'text'
      }));
  };
  
  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (err) {
      return dateString;
    }
  };
  
  // Render loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '300px' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  // Render error state
  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }
  
  // Render no metadata state
  if (!metadata) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="body1" color="textSecondary">
          No metadata available for this file.
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          sx={{ mt: 2 }}
        >
          Refresh
        </Button>
      </Paper>
    );
  }
  
  return (
    <Box>
      {/* Success Message */}
      {saveSuccess && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSaveSuccess(false)}>
          Metadata saved successfully.
        </Alert>
      )}
      
      {/* File Information */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          File Information
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Filename"
              value={metadata.filename || ''}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="File Format"
              value={metadata.file_format || ''}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Created At"
              value={formatDate(metadata.created_at)}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
        </Grid>
      </Paper>
      
      {/* Audio Characteristics */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Audio Characteristics
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Duration (seconds)"
              value={metadata.duration || ''}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Sample Rate (Hz)"
              value={metadata.sample_rate || ''}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Channels"
              value={metadata.channels || ''}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="RMS Energy (mean)"
              value={metadata.rms_mean || ''}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Spectral Centroid (mean)"
              value={metadata.spectral_centroid_mean || ''}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Spectral Bandwidth (mean)"
              value={metadata.spectral_bandwidth_mean || ''}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Zero Crossing Rate (mean)"
              value={metadata.zero_crossing_rate_mean || ''}
              fullWidth
              margin="normal"
              InputProps={{
                readOnly: true,
              }}
            />
          </Grid>
        </Grid>
      </Paper>
      
      {/* Custom Metadata */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Custom Metadata
          </Typography>
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={handleAddCustomMetadata}
            disabled={saving}
          >
            Add Field
          </Button>
        </Box>
        
        {getCustomMetadataFields().length > 0 ? (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Field Name</TableCell>
                  <TableCell>Value</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {getCustomMetadataFields().map((field) => (
                  <TableRow key={field.name} hover>
                    <TableCell>{field.name}</TableCell>
                    <TableCell>{field.value}</TableCell>
                    <TableCell>
                      <Chip label={field.type} size="small" />
                    </TableCell>
                    <TableCell>
                      <Tooltip title="Delete Field">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleDeleteCustomMetadata(field.name)}
                          disabled={saving}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center', py: 2 }}>
            No custom metadata fields. Click "Add Field" to add one.
          </Typography>
        )}
      </Paper>
      
      {/* Actions */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          disabled={loading || saving}
          sx={{ mr: 1 }}
        >
          Refresh
        </Button>
        <Button
          variant="outlined"
          startIcon={<DownloadIcon />}
          onClick={handleExportMetadata}
          disabled={loading || saving}
          sx={{ mr: 1 }}
        >
          Export
        </Button>
        <Button
          variant="contained"
          color="primary"
          startIcon={saving ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />}
          onClick={handleSave}
          disabled={loading || saving}
        >
          Save
        </Button>
      </Box>
      
      {/* Custom Metadata Dialog */}
      <Dialog open={customMetadataDialog.open} onClose={handleCustomMetadataDialogClose}>
        <DialogTitle>Add Custom Metadata Field</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Add a custom metadata field to this file. Field names must be unique.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            label="Field Name"
            fullWidth
            value={customMetadataDialog.fieldName}
            onChange={(e) => handleCustomMetadataDialogChange('fieldName', e.target.value)}
            sx={{ mt: 2 }}
          />
          <TextField
            margin="dense"
            label="Field Value"
            fullWidth
            value={customMetadataDialog.fieldValue}
            onChange={(e) => handleCustomMetadataDialogChange('fieldValue', e.target.value)}
          />
          <FormControl fullWidth margin="dense">
            <InputLabel>Field Type</InputLabel>
            <Select
              value={customMetadataDialog.fieldType}
              onChange={(e) => handleCustomMetadataDialogChange('fieldType', e.target.value)}
              label="Field Type"
            >
              <MenuItem value="text">Text</MenuItem>
              <MenuItem value="number">Number</MenuItem>
              <MenuItem value="boolean">Boolean</MenuItem>
              <MenuItem value="date">Date</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCustomMetadataDialogClose} disabled={saving}>
            Cancel
          </Button>
          <Button 
            onClick={handleCustomMetadataDialogSave} 
            color="primary" 
            disabled={saving || !customMetadataDialog.fieldName}
          >
            {saving ? <CircularProgress size={20} /> : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog.open} onClose={handleDeleteDialogClose}>
        <DialogTitle>Delete Custom Metadata Field</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the field "{deleteDialog.fieldName}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteDialogClose} disabled={saving}>
            Cancel
          </Button>
          <Button 
            onClick={handleDeleteDialogConfirm} 
            color="error" 
            disabled={saving}
          >
            {saving ? <CircularProgress size={20} /> : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MetadataEditor;