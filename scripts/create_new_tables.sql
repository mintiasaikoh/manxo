-- 新しい接続情報テーブル
CREATE TABLE IF NOT EXISTS object_connections (
    id SERIAL PRIMARY KEY,
    source_object_type VARCHAR(100) NOT NULL,
    source_port INTEGER NOT NULL DEFAULT 0,
    target_object_type VARCHAR(100) NOT NULL,
    target_port INTEGER NOT NULL DEFAULT 0,
    patch_file VARCHAR(500) NOT NULL,
    file_type VARCHAR(10) NOT NULL, -- 'maxpat' or 'amxd'
    device_type VARCHAR(50), -- 'midi_effect', 'audio_effect', 'instrument', etc
    hierarchy_depth INTEGER DEFAULT 0,
    parent_context VARCHAR(200), -- 親オブジェクト名（p, gen~など）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- インデックス作成
CREATE INDEX idx_source_object ON object_connections(source_object_type);
CREATE INDEX idx_target_object ON object_connections(target_object_type);
CREATE INDEX idx_patch_file ON object_connections(patch_file);
CREATE INDEX idx_connection_pair ON object_connections(source_object_type, target_object_type);

-- 接続パターン集計用のマテリアライズドビュー
CREATE MATERIALIZED VIEW connection_patterns AS
SELECT 
    source_object_type,
    target_object_type,
    COUNT(*) as occurrence_count,
    COUNT(DISTINCT patch_file) as unique_patches,
    array_agg(DISTINCT file_type) as file_types
FROM object_connections
GROUP BY source_object_type, target_object_type
ORDER BY occurrence_count DESC;

-- ポート使用統計
CREATE MATERIALIZED VIEW port_usage_stats AS
SELECT 
    source_object_type as object_type,
    'output' as port_direction,
    source_port as port_number,
    COUNT(*) as usage_count
FROM object_connections
GROUP BY source_object_type, source_port
UNION ALL
SELECT 
    target_object_type as object_type,
    'input' as port_direction,
    target_port as port_number,
    COUNT(*) as usage_count
FROM object_connections
GROUP BY target_object_type, target_port
ORDER BY object_type, port_direction, port_number;