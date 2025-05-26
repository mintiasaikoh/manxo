#!/usr/bin/env python3
"""
node.scriptとjsオブジェクトの使用状況を分析するスクリプト

このスクリプトは以下の分析を行います：
1. node.scriptで参照されているJavaScriptファイル名のパターンを集計
2. jsオブジェクトで参照されているJavaScriptファイル名のパターンを集計
3. node.scriptの保存属性（autostart、defer、watch等）の設定傾向
4. dependency_cacheに記録されているJavaScriptファイルのパス情報
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from db_connector import DatabaseConnector
import pandas as pd


def extract_js_filename(text):
    """
    textフィールドからJavaScriptファイル名を抽出
    
    パターン例:
    - "node.script example.js"
    - "js myScript.js"
    - "node.script example.js @autostart 1"
    """
    if not text:
        return None
    
    # node.scriptまたはjsの後のファイル名を抽出
    match = re.match(r'^(?:node\.script|js)\s+([^\s@]+\.js)', text)
    if match:
        return match.group(1)
    return None


def extract_node_script_attributes(text):
    """
    node.scriptの属性を抽出
    
    例: "node.script example.js @autostart 1 @defer 0 @watch 1"
    """
    if not text or not text.startswith('node.script'):
        return {}
    
    attributes = {}
    
    # @属性名 値 のパターンを全て抽出
    attr_pattern = r'@(\w+)\s+(\S+)'
    matches = re.findall(attr_pattern, text)
    
    for attr_name, attr_value in matches:
        attributes[attr_name] = attr_value
    
    return attributes


def analyze_dependency_cache(db):
    """dependency_cacheテーブルからJavaScriptファイル情報を分析"""
    
    # dependency_cacheテーブルが存在しないため、スキップ
    return None, Counter(), defaultdict(Counter)


def main():
    """メイン処理"""
    
    # データベース接続
    config_path = Path(__file__).parent / 'db_settings.ini'
    db = DatabaseConnector(str(config_path))
    db.connect()
    
    try:
        print("=== node.scriptとjsオブジェクトの使用状況分析 ===\n")
        
        # 1. node.scriptオブジェクトの分析
        print("1. node.scriptオブジェクトの分析")
        print("-" * 50)
        
        node_script_query = """
        SELECT 
            od.text_content,
            od.saved_attributes,
            COUNT(*) as count
        FROM object_details od
        WHERE od.object_type = 'node.script'
        GROUP BY od.text_content, od.saved_attributes
        ORDER BY count DESC
        """
        
        node_script_results = db.execute_query(node_script_query)
        
        # JavaScriptファイル名を集計
        node_script_files = Counter()
        node_script_attributes = defaultdict(Counter)
        
        for row in node_script_results:
            text = row['text_content']
            saved_attrs = row['saved_attributes']
            count = row['count']
            
            # ファイル名を抽出
            js_file = extract_js_filename(text)
            if js_file:
                node_script_files[js_file] += count
            
            # 属性を抽出
            attrs = extract_node_script_attributes(text)
            for attr_name, attr_value in attrs.items():
                node_script_attributes[attr_name][attr_value] += count
            
            # saved_attributesも解析
            if saved_attrs:
                try:
                    saved_dict = json.loads(saved_attrs)
                    for key, value in saved_dict.items():
                        node_script_attributes[f"saved_{key}"][str(value)] += count
                except:
                    pass
        
        print(f"総node.scriptオブジェクト数: {sum(row['count'] for row in node_script_results)}")
        print(f"ユニークなnode.script設定数: {len(node_script_results)}")
        print(f"\n最も使用されているJavaScriptファイル (Top 20):")
        for js_file, count in node_script_files.most_common(20):
            print(f"  {js_file}: {count}回")
        
        print(f"\nnode.script属性の使用状況:")
        for attr_name, values in sorted(node_script_attributes.items()):
            print(f"\n  @{attr_name}:")
            for value, count in values.most_common(5):
                print(f"    {value}: {count}回")
        
        # 2. jsオブジェクトの分析
        print("\n\n2. jsオブジェクトの分析")
        print("-" * 50)
        
        js_query = """
        SELECT 
            od.text_content,
            COUNT(*) as count
        FROM object_details od
        WHERE od.object_type = 'js'
        GROUP BY od.text_content
        ORDER BY count DESC
        """
        
        js_results = db.execute_query(js_query)
        
        # JavaScriptファイル名を集計
        js_files = Counter()
        js_arguments = Counter()
        
        for row in js_results:
            text = row['text_content']
            count = row['count']
            
            # ファイル名を抽出
            js_file = extract_js_filename(text)
            if js_file:
                js_files[js_file] += count
            
            # 引数も分析
            if text:
                parts = text.split()
                if len(parts) > 2:  # js filename.js arguments...
                    args = ' '.join(parts[2:])
                    js_arguments[args] += count
        
        print(f"総jsオブジェクト数: {sum(row['count'] for row in js_results)}")
        print(f"ユニークなjs設定数: {len(js_results)}")
        print(f"\n最も使用されているJavaScriptファイル (Top 20):")
        for js_file, count in js_files.most_common(20):
            print(f"  {js_file}: {count}回")
        
        if js_arguments:
            print(f"\njs引数パターン (Top 10):")
            for args, count in js_arguments.most_common(10):
                print(f"  {args}: {count}回")
        
        # 3. dependency_cacheの分析
        print("\n\n3. dependency_cacheのJavaScriptファイル分析")
        print("-" * 50)
        
        dep_results, path_depths, path_components = analyze_dependency_cache(db)
        
        if dep_results:
            print(f"dependency_cacheに記録されたJavaScriptファイル数: {len(dep_results)}")
            
            # パスの深さ分布
            print("\nパスの深さ分布:")
            for depth, count in sorted(path_depths.items()):
                print(f"  深さ{depth}: {count}ファイル")
            
            # よく使われるディレクトリ
            print("\nよく使われるディレクトリ (第1階層):")
            if 0 in path_components:
                for dir_name, count in path_components[0].most_common(10):
                    print(f"  {dir_name}: {count}回")
        
        # 4. 統計サマリー
        print("\n\n4. 統計サマリー")
        print("-" * 50)
        
        # ファイル名の傾向を分析
        all_js_files = Counter()
        all_js_files.update(node_script_files)
        all_js_files.update(js_files)
        
        # ファイル名のパターンを分析
        file_patterns = defaultdict(list)
        for filename in all_js_files:
            # 拡張子を除いた名前でパターンを分類
            base_name = filename.replace('.js', '')
            
            # キャメルケース、スネークケース、ケバブケースを判定
            if '_' in base_name:
                file_patterns['snake_case'].append(filename)
            elif '-' in base_name:
                file_patterns['kebab-case'].append(filename)
            elif base_name != base_name.lower() and base_name != base_name.upper():
                file_patterns['camelCase'].append(filename)
            else:
                file_patterns['other'].append(filename)
        
        print("ファイル名の命名規則:")
        for pattern, files in file_patterns.items():
            print(f"  {pattern}: {len(files)}ファイル ({len(files)/len(all_js_files)*100:.1f}%)")
        
        # 共通の接頭辞/接尾辞を分析
        prefixes = Counter()
        suffixes = Counter()
        
        for filename in all_js_files:
            base_name = filename.replace('.js', '')
            parts = re.split(r'[_\-\.]', base_name)
            
            if len(parts) > 1:
                prefixes[parts[0]] += all_js_files[filename]
                suffixes[parts[-1]] += all_js_files[filename]
        
        print("\nよく使われる接頭辞 (Top 10):")
        for prefix, count in prefixes.most_common(10):
            print(f"  {prefix}: {count}回")
        
        print("\nよく使われる接尾辞 (Top 10):")
        for suffix, count in suffixes.most_common(10):
            print(f"  {suffix}: {count}回")
        
        # 5. 詳細レポートをファイルに保存
        output_dir = Path(__file__).parent.parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        report = {
            'node_script_files': dict(node_script_files.most_common()),
            'js_files': dict(js_files.most_common()),
            'node_script_attributes': {k: dict(v) for k, v in node_script_attributes.items()},
            'js_arguments': dict(js_arguments.most_common()),
            'file_patterns': {k: v for k, v in file_patterns.items()},
            'total_stats': {
                'total_node_scripts': sum(row['count'] for row in node_script_results),
                'total_js_objects': sum(row['count'] for row in js_results),
                'unique_js_files': len(all_js_files),
                'dependency_cache_js_files': len(dep_results) if dep_results else 0
            }
        }
        
        report_path = output_dir / 'node_script_js_analysis.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n\n詳細レポートを保存しました: {report_path}")
        
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()