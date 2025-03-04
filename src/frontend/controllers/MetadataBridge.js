/**
 * MetadataBridge.js
 * 
 * This module provides a bridge between the frontend and the backend
 * for metadata management operations.
 */

import PythonBridge from './PythonBridge';

/**
 * MetadataBridge class for metadata management operations
 */
class MetadataBridge {
  /**
   * Get metadata for a file
   * 
   * @param {number} fileId - ID of the file
   * @returns {Promise<object>} - Promise resolving to metadata object
   */
  static async getMetadata(fileId) {
    try {
      const response = await PythonBridge.invoke('metadata.get_metadata', { file_id: fileId });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error getting metadata:', error);
      return {
        success: false,
        error: error.message || 'Failed to get metadata'
      };
    }
  }

  /**
   * Update metadata for a file
   * 
   * @param {number} fileId - ID of the file
   * @param {object} metadata - Metadata object
   * @returns {Promise<object>} - Promise resolving to success status
   */
  static async updateMetadata(fileId, metadata) {
    try {
      const response = await PythonBridge.invoke('metadata.update_metadata', {
        file_id: fileId,
        metadata: metadata
      });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error updating metadata:', error);
      return {
        success: false,
        error: error.message || 'Failed to update metadata'
      };
    }
  }

  /**
   * Add custom metadata for a file
   * 
   * @param {number} fileId - ID of the file
   * @param {string} fieldName - Name of the custom field
   * @param {string} fieldValue - Value for the custom field
   * @param {string} fieldType - Type of the custom field
   * @returns {Promise<object>} - Promise resolving to success status
   */
  static async addCustomMetadata(fileId, fieldName, fieldValue, fieldType = 'text') {
    try {
      const response = await PythonBridge.invoke('metadata.add_custom_metadata', {
        file_id: fileId,
        field_name: fieldName,
        field_value: fieldValue,
        field_type: fieldType
      });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error adding custom metadata:', error);
      return {
        success: false,
        error: error.message || 'Failed to add custom metadata'
      };
    }
  }

  /**
   * Remove custom metadata for a file
   * 
   * @param {number} fileId - ID of the file
   * @param {string} fieldName - Name of the custom field
   * @returns {Promise<object>} - Promise resolving to success status
   */
  static async removeCustomMetadata(fileId, fieldName) {
    try {
      const response = await PythonBridge.invoke('metadata.remove_custom_metadata', {
        file_id: fileId,
        field_name: fieldName
      });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error removing custom metadata:', error);
      return {
        success: false,
        error: error.message || 'Failed to remove custom metadata'
      };
    }
  }

  /**
   * Search for files based on metadata criteria
   * 
   * @param {object} criteria - Search criteria
   * @param {number} limit - Maximum number of results
   * @param {number} offset - Offset for pagination
   * @returns {Promise<object>} - Promise resolving to search results
   */
  static async searchByMetadata(criteria, limit = 100, offset = 0) {
    try {
      const response = await PythonBridge.invoke('metadata.search_by_metadata', {
        criteria: criteria,
        limit: limit,
        offset: offset
      });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error searching metadata:', error);
      return {
        success: false,
        error: error.message || 'Failed to search metadata'
      };
    }
  }

  /**
   * Export metadata for a file
   * 
   * @param {number} fileId - ID of the file
   * @returns {Promise<object>} - Promise resolving to success status
   */
  static async exportMetadata(fileId) {
    try {
      const response = await PythonBridge.invoke('metadata.export_metadata', {
        file_id: fileId
      });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error exporting metadata:', error);
      return {
        success: false,
        error: error.message || 'Failed to export metadata'
      };
    }
  }

  /**
   * Import metadata for a file
   * 
   * @param {number} fileId - ID of the file
   * @param {string} importPath - Path to the metadata file
   * @returns {Promise<object>} - Promise resolving to success status
   */
  static async importMetadata(fileId, importPath) {
    try {
      const response = await PythonBridge.invoke('metadata.import_metadata', {
        file_id: fileId,
        import_path: importPath
      });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error importing metadata:', error);
      return {
        success: false,
        error: error.message || 'Failed to import metadata'
      };
    }
  }

  /**
   * Extract metadata from an audio file
   * 
   * @param {string} filePath - Path to the audio file
   * @param {boolean} extractAudioCharacteristics - Whether to extract audio characteristics
   * @returns {Promise<object>} - Promise resolving to metadata object
   */
  static async extractMetadata(filePath, extractAudioCharacteristics = true) {
    try {
      const response = await PythonBridge.invoke('metadata.extract_metadata', {
        file_path: filePath,
        extract_audio_characteristics: extractAudioCharacteristics
      });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error extracting metadata:', error);
      return {
        success: false,
        error: error.message || 'Failed to extract metadata'
      };
    }
  }

  /**
   * Batch extract metadata for all audio files in a directory
   * 
   * @param {string} directory - Directory to scan for audio files
   * @param {boolean} recursive - Whether to scan subdirectories
   * @returns {Promise<object>} - Promise resolving to batch extraction results
   */
  static async batchExtractMetadata(directory, recursive = false) {
    try {
      const response = await PythonBridge.invoke('metadata.batch_extract_metadata', {
        directory: directory,
        recursive: recursive
      });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error in batch metadata extraction:', error);
      return {
        success: false,
        error: error.message || 'Failed to extract metadata in batch'
      };
    }
  }

  /**
   * Get metadata statistics
   * 
   * @returns {Promise<object>} - Promise resolving to metadata statistics
   */
  static async getMetadataStatistics() {
    try {
      const response = await PythonBridge.invoke('metadata.get_metadata_statistics', {});
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error getting metadata statistics:', error);
      return {
        success: false,
        error: error.message || 'Failed to get metadata statistics'
      };
    }
  }

  /**
   * List files in the database
   * 
   * @param {string} filename - Filter by filename
   * @param {string} fileFormat - Filter by file format
   * @returns {Promise<object>} - Promise resolving to file list
   */
  static async listFiles(filename = null, fileFormat = null) {
    try {
      const response = await PythonBridge.invoke('metadata.list_files', {
        filename: filename,
        file_format: fileFormat
      });
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('Error listing files:', error);
      return {
        success: false,
        error: error.message || 'Failed to list files'
      };
    }
  }
}

export default MetadataBridge;