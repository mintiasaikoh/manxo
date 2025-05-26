-- 処理履歴を追跡するテーブル
CREATE TABLE IF NOT EXISTS process_history (
    id SERIAL PRIMARY KEY,
    patch_file VARCHAR(512) NOT NULL,
    file_hash VARCHAR(64),  -- ファイルの変更を検出するためのハッシュ
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'completed',
    connections_found INTEGER,
    objects_found INTEGER,
    error_message TEXT,
    processing_time_ms INTEGER,
    UNIQUE(patch_file)
);

-- インデックス作成
CREATE INDEX idx_process_history_patch_file ON process_history(patch_file);
CREATE INDEX idx_process_history_processed_at ON process_history(processed_at);

-- 処理統計ビュー
CREATE VIEW processing_stats AS
SELECT 
    DATE(processed_at) as process_date,
    COUNT(*) as files_processed,
    SUM(connections_found) as total_connections,
    SUM(objects_found) as total_objects,
    AVG(processing_time_ms) as avg_time_ms,
    COUNT(CASE WHEN status = 'error' THEN 1 END) as error_count
FROM process_history
GROUP BY DATE(processed_at)
ORDER BY process_date DESC;