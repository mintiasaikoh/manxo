#!/usr/bin/env python3
from db_connector import DatabaseConnector

db = DatabaseConnector('scripts/db_settings.ini')
db.connect()

# インレット数別のポートタイプ定義状況を分析
print('=== インレット数別のポートタイプ定義状況 ===\n')

result = db.execute_query('''
    SELECT 
        num_inlets,
        COUNT(*) as total_objects,
        COUNT(CASE WHEN inlet_types::text = '[]' THEN 1 END) as empty_types,
        COUNT(CASE WHEN inlet_types::text LIKE '%INLET_TYPE%' THEN 1 END) as generic_types,
        COUNT(CASE WHEN inlet_types::text NOT LIKE '%INLET_TYPE%' 
                   AND inlet_types::text != '[]' 
                   AND inlet_types IS NOT NULL THEN 1 END) as specific_types
    FROM objects
    WHERE num_inlets IS NOT NULL
    GROUP BY num_inlets
    ORDER BY num_inlets
    LIMIT 10
''')

print('インレット数 | 総数 | 空[] | 汎用INLET_TYPE | 具体的な型')
print('-' * 60)
for row in result:
    total = row['total_objects']
    empty_pct = row['empty_types'] / total * 100 if total > 0 else 0
    generic_pct = row['generic_types'] / total * 100 if total > 0 else 0
    specific_pct = row['specific_types'] / total * 100 if total > 0 else 0
    
    print(f'{row["num_inlets"]:11d} | {total:4d} | {row["empty_types"]:3d} ({empty_pct:4.1f}%) | {row["generic_types"]:3d} ({generic_pct:4.1f}%) | {row["specific_types"]:3d} ({specific_pct:4.1f}%)')

# インレット1個で具体的な型を持つオブジェクトの例
print('\n\n=== インレット1個で具体的な型を持つオブジェクトの例 ===')
examples = db.execute_query('''
    SELECT object_name, inlet_types, outlet_types
    FROM objects
    WHERE num_inlets = 1
    AND inlet_types::text NOT LIKE '%INLET_TYPE%'
    AND inlet_types::text != '[]'
    AND inlet_types IS NOT NULL
    LIMIT 10
''')

for row in examples:
    print(f'{row["object_name"]}: inlet={row["inlet_types"]}, outlet={row["outlet_types"]}')

# インレット0個のオブジェクト
print('\n\n=== インレット0個のオブジェクトの例 ===')
no_inlet = db.execute_query('''
    SELECT object_name, outlet_types, category
    FROM objects
    WHERE num_inlets = 0
    LIMIT 20
''')

for row in no_inlet:
    print(f'{row["object_name"]:<20} outlet={row["outlet_types"]:<30} (category: {row["category"]})')

# インレット数と型定義の相関を確認
print('\n\n=== 型定義の明確さとインレット数の関係 ===')
correlation = db.execute_query('''
    SELECT 
        CASE 
            WHEN num_inlets = 0 THEN '0個（入力なし）'
            WHEN num_inlets = 1 THEN '1個（単一入力）'
            WHEN num_inlets = 2 THEN '2個（バイナリ演算など）'
            WHEN num_inlets >= 3 THEN '3個以上（複雑）'
        END as inlet_group,
        COUNT(*) as total,
        COUNT(CASE WHEN inlet_types::text NOT LIKE '%INLET_TYPE%' 
                   AND inlet_types::text != '[]' THEN 1 END) as has_specific_type,
        ROUND(100.0 * COUNT(CASE WHEN inlet_types::text NOT LIKE '%INLET_TYPE%' 
                                 AND inlet_types::text != '[]' THEN 1 END) / COUNT(*), 1) as specific_type_pct
    FROM objects
    WHERE num_inlets IS NOT NULL
    GROUP BY inlet_group
    ORDER BY 
        CASE inlet_group
            WHEN '0個（入力なし）' THEN 1
            WHEN '1個（単一入力）' THEN 2
            WHEN '2個（バイナリ演算など）' THEN 3
            ELSE 4
        END
''')

for row in correlation:
    print(f'{row["inlet_group"]:<20} 総数: {row["total"]:4d}, 具体的な型: {row["has_specific_type"]:4d} ({row["specific_type_pct"]}%)')

db.disconnect()