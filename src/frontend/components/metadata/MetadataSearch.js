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
  TablePagination,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import { 
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Clear as ClearIcon,
  Info as InfoIcon,
  AudioFile as AudioFileIcon
} from '@mui/icons-material';
import MetadataBridge from '../../controllers/MetadataBridge';

/**
 * Metadata Search Component
 * 
 * This component provides a user interface for searching files
 * based on metadata criteria.
 */
const MetadataSearch = ({ onFileSelect }) => {
  // State for search criteria
  const [criteria, setCriteria] = useState({
    filename: '',
    file_format: '',
    duration_min: '',
    duration_max: '',
    created_after: '',
    created_before: ''
  });
  
  // State for search results
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // State for pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  
  // State for statistics
  const [statistics, setStatistics] = useState(null);
  const [loadingStats, setLoadingStats] = useState(false);
  
  // Load statistics on component mount
  useEffect(() => {
    loadStatistics();
  }, []);
  
  // Load metadata statistics
  const loadStatistics = async () => {
    setLoadingStats(true);
    setError(null);
    
    try {
      const response = await MetadataBridge.getMetadataStatistics();
      
      if (response.success) {
        setStatistics(response.data || {});
      } else {
        setError(response.error || 'Failed to load metadata statistics');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while loading metadata statistics');
    } finally {
      setLoadingStats(false);
    }
  };
  
  // Handle search criteria change
  const handleCriteriaChange = (field, value) => {
    setCriteria(prevCriteria => ({
      ...prevCriteria,
      [field]: value
    }));
  };
  
  // Handle search button click
  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    setResults(null);
    
    try {
      // Filter out empty criteria
      const filteredCriteria = Object.entries(criteria)
        .filter(([_, value]) => value !== '')
        .reduce((obj, [key, value]) => {
          obj[key] = value;
          return obj;
        }, {});
      
      const response = await MetadataBridge.searchByMetadata(
        filteredCriteria,
        rowsPerPage,
        page * rowsPerPage
      );
      
      if (response.success) {
        setResults(response.data || { data: [] });
      } else {
        setError(response.error || 'Failed to search metadata');
        setResults(null);
      }
    } catch (err) {
      setError(err.message || 'An error occurred while searching metadata');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle clear button click
  const handleClear = () => {
    setCriteria({
      filename: '',
      file_format: '',
      duration_min: '',
      duration_max: '',
      created_after: '',
      created_before: ''
    });
    setResults(null);
    setError(null);
  };
  
  // Handle page change
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
    
    // If we have results, search again with new page
    if (results) {
      handleSearch();
    }
  };
  
  // Handle rows per page change
  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
    
    // If we have results, search again with new rows per page
    if (results) {
      handleSearch();
    }
  };
  
  // Handle file selection
  const handleFileSelect = (fileId) => {
    if (onFileSelect) {
      onFileSelect(fileId);
    }
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
  
  // Format duration
  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };
  
  return (
    <Box>
      {/* Error Message */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {/* Statistics */}
      <Accordion sx={{ mb: 2 }}>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="metadata-statistics-content"
          id="metadata-statistics-header"
        >
          <Typography variant="h6">Metadata Statistics</Typography>
        </AccordionSummary>
        <AccordionDetails>
          {loadingStats ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress size={24} />
            </Box>
          ) : statistics ? (
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h6" color="primary">
                    {statistics.total_files || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Total Files
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h6" color="primary">
                    {statistics.files_with_metadata || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Files with Metadata
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h6" color="primary">
                    {statistics.files_with_custom_metadata || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Files with Custom Metadata
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h6" color="primary">
                    {statistics.custom_fields?.length || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Custom Field Types
                  </Typography>
                </Paper>
              </Grid>
              
              {/* Format Distribution */}
              {statistics.format_distribution && Object.keys(statistics.format_distribution).length > 0 && (
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Format Distribution
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {Object.entries(statistics.format_distribution).map(([format, count]) => (
                      <Chip
                        key={format}
                        label={`${format}: ${count}`}
                        color="primary"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Grid>
              )}
              
              {/* Duration Statistics */}
              {statistics.duration_stats && (
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Duration Statistics
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <Paper sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="body2" color="textSecondary">
                          Min
                        </Typography>
                        <Typography variant="body1">
                          {formatDuration(statistics.duration_stats.min)}
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={4}>
                      <Paper sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="body2" color="textSecondary">
                          Max
                        </Typography>
                        <Typography variant="body1">
                          {formatDuration(statistics.duration_stats.max)}
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={4}>
                      <Paper sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="body2" color="textSecondary">
                          Average
                        </Typography>
                        <Typography variant="body1">
                          {formatDuration(statistics.duration_stats.avg)}
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                </Grid>
              )}
              
              {/* Custom Fields */}
              {statistics.custom_fields && statistics.custom_fields.length > 0 && (
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Custom Fields
                  </Typography>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Field Name</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Count</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {statistics.custom_fields.map((field) => (
                          <TableRow key={field.name}>
                            <TableCell>{field.name}</TableCell>
                            <TableCell>{field.type}</TableCell>
                            <TableCell>{field.count}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              )}
            </Grid>
          ) : (
            <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center' }}>
              No statistics available.
            </Typography>
          )}
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button
              startIcon={<RefreshIcon />}
              onClick={loadStatistics}
              disabled={loadingStats}
            >
              Refresh Statistics
            </Button>
          </Box>
        </AccordionDetails>
      </Accordion>
      
      {/* Search Form */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Search Criteria
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Filename"
              value={criteria.filename}
              onChange={(e) => handleCriteriaChange('filename', e.target.value)}
              fullWidth
              margin="normal"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth margin="normal">
              <InputLabel>File Format</InputLabel>
              <Select
                value={criteria.file_format}
                onChange={(e) => handleCriteriaChange('file_format', e.target.value)}
                label="File Format"
              >
                <MenuItem value="">Any</MenuItem>
                {statistics?.format_distribution && Object.keys(statistics.format_distribution).map((format) => (
                  <MenuItem key={format} value={format}>{format}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Minimum Duration (seconds)"
              type="number"
              value={criteria.duration_min}
              onChange={(e) => handleCriteriaChange('duration_min', e.target.value)}
              fullWidth
              margin="normal"
              inputProps={{ min: 0 }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Maximum Duration (seconds)"
              type="number"
              value={criteria.duration_max}
              onChange={(e) => handleCriteriaChange('duration_max', e.target.value)}
              fullWidth
              margin="normal"
              inputProps={{ min: 0 }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Created After"
              type="datetime-local"
              value={criteria.created_after}
              onChange={(e) => handleCriteriaChange('created_after', e.target.value)}
              fullWidth
              margin="normal"
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Created Before"
              type="datetime-local"
              value={criteria.created_before}
              onChange={(e) => handleCriteriaChange('created_before', e.target.value)}
              fullWidth
              margin="normal"
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
        </Grid>
        
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
          <Button
            variant="outlined"
            startIcon={<ClearIcon />}
            onClick={handleClear}
            sx={{ mr: 1 }}
          >
            Clear
          </Button>
          <Button
            variant="contained"
            color="primary"
            startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
            onClick={handleSearch}
            disabled={loading}
          >
            Search
          </Button>
        </Box>
      </Paper>
      
      {/* Search Results */}
      {results && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Search Results
          </Typography>
          
          {results.data && results.data.length > 0 ? (
            <>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Filename</TableCell>
                      <TableCell>Format</TableCell>
                      <TableCell>Duration</TableCell>
                      <TableCell>Created At</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {results.data.map((file) => (
                      <TableRow key={file.file_id} hover>
                        <TableCell>{file.filename}</TableCell>
                        <TableCell>{file.file_format}</TableCell>
                        <TableCell>{formatDuration(file.duration)}</TableCell>
                        <TableCell>{formatDate(file.created_at)}</TableCell>
                        <TableCell>
                          <Tooltip title="View/Edit Metadata">
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => handleFileSelect(file.file_id)}
                            >
                              <InfoIcon />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              <TablePagination
                component="div"
                count={results.total_count || 0}
                page={page}
                onPageChange={handleChangePage}
                rowsPerPage={rowsPerPage}
                onRowsPerPageChange={handleChangeRowsPerPage}
                rowsPerPageOptions={[5, 10, 25, 50]}
              />
            </>
          ) : (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <AudioFileIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="body1" color="textSecondary">
                No files found matching your search criteria.
              </Typography>
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
};

export default MetadataSearch;