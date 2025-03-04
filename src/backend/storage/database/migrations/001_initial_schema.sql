-- Migration: Initial Schema
-- Created: 2025-03-04 06:59:00

-- This migration creates the initial database schema for the Voices application.
-- It includes tables for speakers, processed files, processing history, and model performance metrics.

-- Note: The actual tables are created by SQLAlchemy's create_all() method,
-- so this migration is just a placeholder to mark the initial schema version.

-- Create schema_migrations table if it doesn't exist
CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version VARCHAR(50) NOT NULL UNIQUE,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description VARCHAR(255)
);

-- Insert record for this migration
INSERT INTO schema_migrations (version, description)
VALUES ('001', 'initial schema');