-- Migration: 002_metadata_tables.sql
-- Description: Add metadata tables for audio files

-- Create file_metadata table
CREATE TABLE IF NOT EXISTS file_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL UNIQUE,
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
    mfcc_means TEXT,  -- JSON string
    mfcc_stds TEXT,   -- JSON string
    extracted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES processed_files (id) ON DELETE CASCADE
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
    FOREIGN KEY (file_id) REFERENCES processed_files (id) ON DELETE CASCADE,
    UNIQUE (file_id, field_name)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_file_metadata_file_id ON file_metadata (file_id);
CREATE INDEX IF NOT EXISTS idx_custom_metadata_file_id ON custom_metadata (file_id);
CREATE INDEX IF NOT EXISTS idx_custom_metadata_field_name ON custom_metadata (field_name);

-- Update migrations table
INSERT INTO migrations (version, description, applied_at)
VALUES (2, 'Add metadata tables for audio files', CURRENT_TIMESTAMP);