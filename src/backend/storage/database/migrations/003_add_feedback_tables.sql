-- Migration: 003_add_feedback_tables.sql
-- Description: Adds tables for the User Feedback Collection System
-- Date: 2025-03-04

-- Create feedback_categories table
CREATE TABLE IF NOT EXISTS feedback_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default feedback categories
INSERT INTO feedback_categories (name, description) VALUES 
    ('Usability', 'Feedback related to the user interface and user experience'),
    ('Quality', 'Feedback related to the quality of voice separation'),
    ('Performance', 'Feedback related to application performance and resource usage'),
    ('Feature Request', 'Suggestions for new features or improvements'),
    ('Bug Report', 'Reports of issues or unexpected behavior');

-- Create feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,  -- Optional, can be NULL for anonymous feedback
    category_id INTEGER,
    title TEXT NOT NULL,
    description TEXT,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),  -- 1-5 star rating
    is_anonymous BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES feedback_categories(id)
);

-- Create feedback_context table
CREATE TABLE IF NOT EXISTS feedback_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id INTEGER NOT NULL,
    context_type TEXT NOT NULL,  -- 'feature', 'model', 'file', etc.
    context_id TEXT,  -- ID of the related entity (feature name, model ID, file ID, etc.)
    context_data TEXT,  -- JSON data with additional context information
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (feedback_id) REFERENCES feedback(id) ON DELETE CASCADE
);

-- Create feedback_tags table
CREATE TABLE IF NOT EXISTS feedback_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create feedback_tag_mapping table (many-to-many relationship)
CREATE TABLE IF NOT EXISTS feedback_tag_mapping (
    feedback_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (feedback_id, tag_id),
    FOREIGN KEY (feedback_id) REFERENCES feedback(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES feedback_tags(id) ON DELETE CASCADE
);

-- Create feedback_analytics table
CREATE TABLE IF NOT EXISTS feedback_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    average_rating REAL,
    positive_count INTEGER DEFAULT 0,  -- Ratings 4-5
    neutral_count INTEGER DEFAULT 0,   -- Rating 3
    negative_count INTEGER DEFAULT 0,  -- Ratings 1-2
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES feedback_categories(id)
);

-- Create feedback_feature_requests table
CREATE TABLE IF NOT EXISTS feedback_feature_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id INTEGER NOT NULL,
    status TEXT DEFAULT 'new',  -- 'new', 'under_review', 'planned', 'in_progress', 'completed', 'declined'
    priority TEXT DEFAULT 'medium',  -- 'low', 'medium', 'high', 'critical'
    votes INTEGER DEFAULT 0,
    assigned_to TEXT,
    planned_release TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (feedback_id) REFERENCES feedback(id) ON DELETE CASCADE
);

-- Create feedback_responses table
CREATE TABLE IF NOT EXISTS feedback_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id INTEGER NOT NULL,
    response_text TEXT NOT NULL,
    responded_by TEXT,
    is_public BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (feedback_id) REFERENCES feedback(id) ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_feedback_category ON feedback(category_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_context_type ON feedback_context(context_type);
CREATE INDEX IF NOT EXISTS idx_feedback_context_id ON feedback_context(context_id);
CREATE INDEX IF NOT EXISTS idx_feedback_feature_requests_status ON feedback_feature_requests(status);
CREATE INDEX IF NOT EXISTS idx_feedback_feature_requests_priority ON feedback_feature_requests(priority);