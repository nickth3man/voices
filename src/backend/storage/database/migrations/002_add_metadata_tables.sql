-- Migration: Add metadata tables
-- This migration adds tables for storing metadata for audio files

-- Create file_metadata table
CREATE TABLE IF NOT EXISTS file_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    duration REAL,
    sample_rate INTEGER,
    channels INTEGER,
    file_format TEXT,
    rms_mean REAL,
    rms_std REAL,
    spectral_centroid_mean REAL,
    spectral_centroid_std REAL,
    spectral_bandwidth_mean REAL,
    spectral_bandwidth_std REAL,
    zero_crossing_rate_mean REAL,
    zero_crossing_rate_std REAL,
    mfcc_means TEXT,  -- JSON array
    mfcc_stds TEXT,   -- JSON array
    extracted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES processed_files(id) ON DELETE CASCADE,
    UNIQUE (file_id)
);

-- Create custom_metadata table
CREATE TABLE IF NOT EXISTS custom_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    field_name TEXT NOT NULL,
    field_value TEXT,
    field_type TEXT NOT NULL DEFAULT 'text',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES processed_files(id) ON DELETE CASCADE,
    UNIQUE (file_id, field_name)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_file_metadata_file_id ON file_metadata(file_id);
CREATE INDEX IF NOT EXISTS idx_custom_metadata_file_id ON custom_metadata(file_id);
CREATE INDEX IF NOT EXISTS idx_custom_metadata_field_name ON custom_metadata(field_name);

-- Add triggers to update the updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_file_metadata_timestamp
AFTER UPDATE ON file_metadata
BEGIN
    UPDATE file_metadata SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_custom_metadata_timestamp
AFTER UPDATE ON custom_metadata
BEGIN
    UPDATE custom_metadata SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;