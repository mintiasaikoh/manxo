-- Max/MSPパッチ分析用の拡張テーブル構造
-- ポートタイプ情報と位置情報を含む

-- 既存のobject_connectionsテーブルに新しいカラムを追加
ALTER TABLE object_connections 
ADD COLUMN IF NOT EXISTS source_outlet_types TEXT[],
ADD COLUMN IF NOT EXISTS target_inlet_types TEXT[],
ADD COLUMN IF NOT EXISTS source_position FLOAT[],  -- [x, y, width, height]
ADD COLUMN IF NOT EXISTS target_position FLOAT[],
ADD COLUMN IF NOT EXISTS connection_order INTEGER;

-- オブジェクト詳細情報を格納する新しいテーブル
CREATE TABLE IF NOT EXISTS object_details (
    id SERIAL PRIMARY KEY,
    patch_file VARCHAR(500),
    object_id VARCHAR(255),
    full_object_id VARCHAR(500),  -- 階層を含む完全ID (例: obj-1:obj-2)
    maxclass VARCHAR(100),
    object_type VARCHAR(100),     -- extract_object_value()から取得したタイプ
    text_content TEXT,            -- textフィールドの完全な内容
    position FLOAT[],             -- [x, y, width, height]
    inlet_types TEXT[],           -- ["", "int", "float", ...]
    outlet_types TEXT[],          -- ["bang", "int", "float", "list", ...]
    fontname VARCHAR(100),
    fontsize FLOAT,
    parameter_enable BOOLEAN,
    numinlets INTEGER,
    numoutlets INTEGER,
    comment TEXT,
    hint TEXT,
    presentation BOOLEAN,
    presentation_rect FLOAT[],    -- プレゼンテーションモードでの位置
    saved_attributes JSONB,       -- その他の属性をJSON形式で保存
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(patch_file, full_object_id)
);

-- インデックスの作成
CREATE INDEX IF NOT EXISTS idx_object_details_patch_file ON object_details(patch_file);
CREATE INDEX IF NOT EXISTS idx_object_details_object_id ON object_details(object_id);
CREATE INDEX IF NOT EXISTS idx_object_details_full_object_id ON object_details(full_object_id);
CREATE INDEX IF NOT EXISTS idx_object_details_maxclass ON object_details(maxclass);
CREATE INDEX IF NOT EXISTS idx_object_details_object_type ON object_details(object_type);

-- 接続パターン分析用のビュー
CREATE OR REPLACE VIEW connection_analysis_view AS
SELECT 
    oc.id,
    oc.patch_file,
    oc.source_object_type,
    oc.source_port,
    oc.source_outlet_types,
    oc.target_object_type,
    oc.target_port,
    oc.target_inlet_types,
    oc.source_value,
    oc.target_value,
    oc.hierarchy_depth,
    oc.parent_context,
    -- 型の互換性チェック
    CASE 
        WHEN oc.source_outlet_types[oc.source_port + 1] = oc.target_inlet_types[oc.target_port + 1] THEN 'exact_match'
        WHEN oc.source_outlet_types[oc.source_port + 1] = 'list' AND oc.target_inlet_types[oc.target_port + 1] = 'float' THEN 'list_to_float'
        WHEN oc.source_outlet_types[oc.source_port + 1] = 'float' AND oc.target_inlet_types[oc.target_port + 1] = 'int' THEN 'float_to_int'
        WHEN oc.source_outlet_types[oc.source_port + 1] IS NULL OR oc.target_inlet_types[oc.target_port + 1] IS NULL THEN 'unknown'
        ELSE 'type_mismatch'
    END as type_compatibility,
    -- 接続距離の計算（マンハッタン距離）
    ABS(oc.source_position[1] - oc.target_position[1]) + ABS(oc.source_position[2] - oc.target_position[2]) as connection_distance
FROM object_connections oc;

-- 空間的なクラスタリング分析用のビュー
CREATE OR REPLACE VIEW spatial_clustering_view AS
SELECT 
    od1.patch_file,
    od1.object_id as object1_id,
    od1.object_type as object1_type,
    od2.object_id as object2_id,
    od2.object_type as object2_type,
    -- ユークリッド距離
    SQRT(POWER(od1.position[1] - od2.position[1], 2) + POWER(od1.position[2] - od2.position[2], 2)) as euclidean_distance,
    -- 中心点
    ARRAY[(od1.position[1] + od2.position[1]) / 2, (od1.position[2] + od2.position[2]) / 2] as center_point
FROM object_details od1
JOIN object_details od2 ON od1.patch_file = od2.patch_file AND od1.object_id < od2.object_id
WHERE od1.position IS NOT NULL AND od2.position IS NOT NULL;