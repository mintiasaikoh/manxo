#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gen関係のオブジェクトを適切に分類してカテゴリを更新
Gen言語で使用されるオペレーターや関数を明確に分類する
"""

import sys
import configparser
from db_connector import DatabaseConnector

def update_gen_objects():
    """Gen関係のオブジェクトのカテゴリを更新"""
    
    db = DatabaseConnector('scripts/db_settings.ini')
    
    try:
        db.connect()
        
        print("=== Gen関係オブジェクトの分類更新開始 ===")
        
        # Gen言語で使用される数学オペレーター
        gen_math_operators = [
            'add', 'sub', 'mul', 'div', 'mod', 'neg', 
            'rsub', 'rdiv', 'rmod', 'absdiff'
        ]
        
        # Gen言語で使用される比較オペレーター
        gen_comparison_operators = [
            'eq', 'eqp', 'gt', 'gte', 'gtp', 'gtep',
            'lt', 'lte', 'ltp', 'ltep', 'neq', 'neqp'
        ]
        
        # Gen言語で使用される論理オペレーター
        gen_logic_operators = [
            'and', 'or', 'xor', 'not', 'bool'
        ]
        
        # Gen言語で使用される数学関数
        gen_math_functions = [
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
            'sinh', 'cosh', 'tanh', 'fastsin', 'fastcos', 'fasttan',
            'exp', 'exp2', 'ln', 'log', 'log2', 'log10', 'fastexp', 'fastpow',
            'pow', 'sqrt', 'abs', 'sign', 'floor', 'ceil', 'fract', 'trunc',
            'min', 'max', 'clamp', 'fold', 'wrap', 'hypot'
        ]
        
        # Gen言語で使用される定数
        gen_constants = [
            'pi', 'halfpi', 'twopi', 'invpi', 'e', 'ln2', 'ln10', 'log2e', 'log10e',
            'sqrt2', 'invsqrt2', 'degtorad', 'radtodeg'
        ]
        
        # Gen言語で使用されるベクター関数
        gen_vector_functions = [
            'dot', 'cross', 'normalize', 'length', 'distance', 'reflect', 'refract',
            'faceforward', 'mix', 'smoothstep', 'step', 'concat', 'swiz', 'vec'
        ]
        
        # Gen言語で使用されるサンプリング関数
        gen_sampling_functions = [
            'sample', 'peek', 'poke', 'lookup', 'nearest', 'linear', 'cubic',
            'spline', 'samplepix', 'nearestpix'
        ]
        
        # Gen言語で使用されるフィルター関数
        gen_filter_functions = [
            'delta', 'sah', 'latch', 'history', 'dcblock', 'phasewrap', 'interp'
        ]
        
        # Gen言語で使用される波形生成関数
        gen_waveform_functions = [
            'noise', 'phasor', 'cycle', 'triangle', 'saw', 'rect', 'train', 'rate'
        ]
        
        # Gen言語で使用される数値変換関数
        gen_conversion_functions = [
            'mstosamps', 'sampstoms', 'ftom', 'mtof', 'atodb', 'dbtoa'
        ]
        
        # 各カテゴリを更新
        update_counts = {}
        
        categories = [
            ('Gen Math Operators', gen_math_operators),
            ('Gen Comparison Operators', gen_comparison_operators), 
            ('Gen Logic Operators', gen_logic_operators),
            ('Gen Math Functions', gen_math_functions),
            ('Gen Constants', gen_constants),
            ('Gen Vector Functions', gen_vector_functions),
            ('Gen Sampling Functions', gen_sampling_functions),
            ('Gen Filter Functions', gen_filter_functions),
            ('Gen Waveform Functions', gen_waveform_functions),
            ('Gen Conversion Functions', gen_conversion_functions)
        ]
        
        for category_name, object_list in categories:
            if not object_list:
                continue
                
            # IN句用のプレースホルダを作成
            placeholders = ','.join(['%s'] * len(object_list))
            query = f"""
            UPDATE objects 
            SET category = %s 
            WHERE object_name IN ({placeholders})
            AND (num_inlets = 0 AND num_outlets = 0)
            """
            
            try:
                cursor = db.connection.cursor()
                cursor.execute(query, [category_name] + object_list)
                updated_count = cursor.rowcount
                update_counts[category_name] = updated_count
                cursor.close()
                db.connection.commit()
                
                if updated_count > 0:
                    print(f"更新: {category_name} -> {updated_count}個のオブジェクト")
                    
            except Exception as e:
                print(f"更新エラー {category_name}: {e}")
        
        # 特別なGenオブジェクトも更新
        special_gen_objects = {
            'codebox': 'Gen Code Container',
            'gen~': 'Gen Audio Processing',
            'gen': 'Gen Data Processing',
            'jit.gen': 'Gen Video Processing'
        }
        
        for obj_name, category in special_gen_objects.items():
            query = "UPDATE objects SET category = %s WHERE object_name = %s"
            try:
                cursor = db.connection.cursor()
                cursor.execute(query, (category, obj_name))
                if cursor.rowcount > 0:
                    print(f"更新: {obj_name} -> {category}")
                cursor.close()
                db.connection.commit()
            except Exception as e:
                print(f"更新エラー {obj_name}: {e}")
        
        # 結果確認
        print(f"\n=== 更新結果サマリー ===")
        total_updated = sum(update_counts.values())
        print(f"Gen関係オブジェクト更新数: {total_updated}個")
        
        for category, count in update_counts.items():
            if count > 0:
                print(f"  {category}: {count}個")
        
        # 最終統計
        query = """
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN num_inlets > 0 OR num_outlets > 0 THEN 1 END) as with_ports,
            COUNT(CASE WHEN num_inlets = 0 AND num_outlets = 0 THEN 1 END) as without_ports,
            COUNT(CASE WHEN category LIKE 'Gen %' THEN 1 END) as gen_objects
        FROM objects
        """
        result = db.execute_query(query)
        stats = result[0]
        
        print(f"\n=== 最終統計 ===")
        print(f"総オブジェクト数: {stats['total']}")
        print(f"ポート情報あり: {stats['with_ports']}")
        print(f"ポート情報なし: {stats['without_ports']}")
        print(f"Gen関係オブジェクト: {stats['gen_objects']}")
        print(f"ポート情報カバー率: {stats['with_ports']/stats['total']*100:.1f}%")
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    update_gen_objects()