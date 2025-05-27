-- MANXO Database Schema
-- PostgreSQL tables for Max/MSP patch analysis

-- Main connection table
CREATE TABLE IF NOT EXISTS object_connections (
    id SERIAL PRIMARY KEY,
    source_object_type VARCHAR(255) NOT NULL,
    source_port INTEGER NOT NULL DEFAULT 0,
    target_object_type VARCHAR(255) NOT NULL,
    target_port INTEGER NOT NULL DEFAULT 0,
    patch_file VARCHAR(500) NOT NULL,
    file_type VARCHAR(10) NOT NULL CHECK (file_type IN ('maxpat', 'amxd')),
    device_type VARCHAR(50),
    hierarchy_depth INTEGER DEFAULT 0,
    parent_context VARCHAR(255),
    source_value TEXT,
    target_value TEXT,
    source_outlet_types TEXT[],
    target_inlet_types TEXT[],
    source_position FLOAT[],
    target_position FLOAT[],
    connection_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_object_type, source_port, target_object_type, target_port, patch_file)
);

-- Object details table
CREATE TABLE IF NOT EXISTS object_details (
    id SERIAL PRIMARY KEY,
    patch_file VARCHAR(500) NOT NULL,
    object_id VARCHAR(100) NOT NULL,
    full_object_id VARCHAR(255) NOT NULL,
    maxclass VARCHAR(100) NOT NULL,
    object_type VARCHAR(255),
    text_content TEXT,
    position FLOAT[],
    inlet_types TEXT[],
    outlet_types TEXT[],
    saved_attributes JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(patch_file, object_id)
);

-- Object type master table
CREATE TABLE IF NOT EXISTS object_types (
    id SERIAL PRIMARY KEY,
    object_name VARCHAR(255) UNIQUE NOT NULL,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    description TEXT,
    inlet_count INTEGER,
    outlet_count INTEGER,
    inlet_types TEXT[],
    outlet_types TEXT[],
    is_ui_object BOOLEAN DEFAULT FALSE,
    is_msp_object BOOLEAN DEFAULT FALSE,
    is_jitter_object BOOLEAN DEFAULT FALSE,
    max_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing status table
CREATE TABLE IF NOT EXISTS processing_status (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(500) UNIQUE NOT NULL,
    file_type VARCHAR(10),
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    processing_time_ms INTEGER,
    object_count INTEGER,
    connection_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_connections_patch ON object_connections(patch_file);
CREATE INDEX IF NOT EXISTS idx_connections_source ON object_connections(source_object_type);
CREATE INDEX IF NOT EXISTS idx_connections_target ON object_connections(target_object_type);
CREATE INDEX IF NOT EXISTS idx_details_patch ON object_details(patch_file);
CREATE INDEX IF NOT EXISTS idx_details_type ON object_details(object_type);
CREATE INDEX IF NOT EXISTS idx_status_file ON processing_status(file_path);

-- Create view for connection patterns
CREATE OR REPLACE VIEW connection_patterns AS
SELECT 
    source_object_type,
    target_object_type,
    COUNT(*) as frequency,
    array_agg(DISTINCT patch_file) as example_patches
FROM object_connections
GROUP BY source_object_type, target_object_type
ORDER BY frequency DESC;