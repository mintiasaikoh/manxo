#!/usr/bin/env python3
"""
Max/MSPパッチ接続データの検証スクリプト

データベース内のobject_connectionsテーブルのデータ整合性を検証し、
問題を検出して報告します。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import re

from db_connector import DatabaseConnector


class DataValidator:
    """接続データの検証とレポート生成"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: データベース設定ファイルのパス
        """
        self.db = DatabaseConnector(config_path)
        self.issues = []
        
    def validate_all(self) -> Dict[str, Any]:
        """すべての検証を実行"""
        print("データ検証を開始します...")
        
        results = {
            'total_connections': 0,
            'validation_results': {},
            'issues': []
        }
        
        try:
            self.db.connect()
            
            # 基本統計
            results['total_connections'] = self._get_total_connections()
            
            # 各種検証を実行
            print("\n1. 接続の整合性を検証中...")
            results['validation_results']['connection_integrity'] = self._validate_connection_integrity()
            
            print("\n2. オブジェクト名の妥当性を検証中...")
            results['validation_results']['object_names'] = self._validate_object_names()
            
            print("\n3. ポート番号の妥当性を検証中...")
            results['validation_results']['port_numbers'] = self._validate_port_numbers()
            
            print("\n4. 値の形式を検証中...")
            results['validation_results']['value_formats'] = self._validate_value_formats()
            
            print("\n5. 階層構造の整合性を検証中...")
            results['validation_results']['hierarchy'] = self._validate_hierarchy()
            
            print("\n6. 重複接続を検証中...")
            results['validation_results']['duplicates'] = self._validate_duplicates()
            
            print("\n7. Gen~内部オブジェクトを検証中...")
            results['validation_results']['gen_objects'] = self._validate_gen_objects()
            
            results['issues'] = self.issues
            
        finally:
            self.db.disconnect()
            
        return results
    
    def _get_total_connections(self) -> int:
        """総接続数を取得"""
        query = "SELECT COUNT(*) as count FROM object_connections"
        result = self.db.execute_query(query)
        return result[0]['count'] if result else 0
    
    def _validate_connection_integrity(self) -> Dict[str, Any]:
        """接続の整合性を検証"""
        results = {
            'total_checked': 0,
            'issues_found': 0,
            'issue_types': defaultdict(int)
        }
        
        # 自己参照をチェック
        query = """
        SELECT COUNT(*) as count
        FROM object_connections
        WHERE source_object_name = target_object_name
        AND source_object_id = target_object_id
        """
        result = self.db.execute_query(query)
        self_references = result[0]['count'] if result else 0
        
        if self_references > 0:
            results['issues_found'] += self_references
            results['issue_types']['self_reference'] = self_references
            self.issues.append({
                'type': 'self_reference',
                'count': self_references,
                'severity': 'warning',
                'description': f'{self_references}個の自己参照接続が見つかりました'
            })
        
        # 同一接続の複数ポートをチェック
        query = """
        SELECT source_object_name, source_object_id, target_object_name, target_object_id,
               COUNT(*) as connection_count
        FROM object_connections
        GROUP BY source_object_name, source_object_id, target_object_name, target_object_id
        HAVING COUNT(*) > 1
        """
        multi_connections = self.db.execute_query(query)
        
        if multi_connections:
            total_multi = sum(row['connection_count'] for row in multi_connections)
            results['issues_found'] += len(multi_connections)
            results['issue_types']['multiple_connections'] = len(multi_connections)
            self.issues.append({
                'type': 'multiple_connections',
                'count': len(multi_connections),
                'severity': 'info',
                'description': f'{len(multi_connections)}組のオブジェクトペアで複数接続があります',
                'details': multi_connections[:5]  # 最初の5件のみ
            })
        
        results['total_checked'] = self._get_total_connections()
        return results
    
    def _validate_object_names(self) -> Dict[str, Any]:
        """オブジェクト名の妥当性を検証"""
        results = {
            'total_checked': 0,
            'issues_found': 0,
            'issue_types': defaultdict(int)
        }
        
        # NULL/空のオブジェクト名をチェック
        query = """
        SELECT COUNT(*) as count
        FROM object_connections
        WHERE source_object_name IS NULL OR source_object_name = ''
        OR target_object_name IS NULL OR target_object_name = ''
        """
        result = self.db.execute_query(query)
        null_names = result[0]['count'] if result else 0
        
        if null_names > 0:
            results['issues_found'] += null_names
            results['issue_types']['null_or_empty_name'] = null_names
            self.issues.append({
                'type': 'null_or_empty_name',
                'count': null_names,
                'severity': 'error',
                'description': f'{null_names}個の接続でオブジェクト名がNULLまたは空です'
            })
        
        # 不正な文字を含むオブジェクト名をチェック
        query = """
        SELECT DISTINCT source_object_name as object_name
        FROM object_connections
        WHERE source_object_name ~ '[<>"|?*]'
        UNION
        SELECT DISTINCT target_object_name as object_name
        FROM object_connections
        WHERE target_object_name ~ '[<>"|?*]'
        """
        invalid_chars = self.db.execute_query(query)
        
        if invalid_chars:
            results['issues_found'] += len(invalid_chars)
            results['issue_types']['invalid_characters'] = len(invalid_chars)
            self.issues.append({
                'type': 'invalid_characters',
                'count': len(invalid_chars),
                'severity': 'warning',
                'description': f'{len(invalid_chars)}個のオブジェクト名に不正な文字が含まれています',
                'examples': [row['object_name'] for row in invalid_chars[:5]]
            })
        
        # 異常に長いオブジェクト名をチェック
        query = """
        SELECT DISTINCT source_object_name as object_name
        FROM object_connections
        WHERE LENGTH(source_object_name) > 100
        UNION
        SELECT DISTINCT target_object_name as object_name
        FROM object_connections
        WHERE LENGTH(target_object_name) > 100
        """
        long_names = self.db.execute_query(query)
        
        if long_names:
            results['issues_found'] += len(long_names)
            results['issue_types']['excessively_long_name'] = len(long_names)
            self.issues.append({
                'type': 'excessively_long_name',
                'count': len(long_names),
                'severity': 'warning',
                'description': f'{len(long_names)}個のオブジェクト名が異常に長いです（100文字以上）',
                'examples': [row['object_name'][:50] + '...' for row in long_names[:3]]
            })
        
        results['total_checked'] = self._get_total_connections() * 2  # source + target
        return results
    
    def _validate_port_numbers(self) -> Dict[str, Any]:
        """ポート番号の妥当性を検証"""
        results = {
            'total_checked': 0,
            'issues_found': 0,
            'issue_types': defaultdict(int)
        }
        
        # 負のポート番号をチェック
        query = """
        SELECT COUNT(*) as count
        FROM object_connections
        WHERE source_port < 0 OR target_port < 0
        """
        result = self.db.execute_query(query)
        negative_ports = result[0]['count'] if result else 0
        
        if negative_ports > 0:
            results['issues_found'] += negative_ports
            results['issue_types']['negative_port'] = negative_ports
            self.issues.append({
                'type': 'negative_port',
                'count': negative_ports,
                'severity': 'error',
                'description': f'{negative_ports}個の接続で負のポート番号が使用されています'
            })
        
        # 異常に大きいポート番号をチェック（通常は0-31程度）
        query = """
        SELECT COUNT(*) as count
        FROM object_connections
        WHERE source_port > 100 OR target_port > 100
        """
        result = self.db.execute_query(query)
        large_ports = result[0]['count'] if result else 0
        
        if large_ports > 0:
            results['issues_found'] += large_ports
            results['issue_types']['excessively_large_port'] = large_ports
            self.issues.append({
                'type': 'excessively_large_port',
                'count': large_ports,
                'severity': 'warning',
                'description': f'{large_ports}個の接続で異常に大きいポート番号（>100）が使用されています'
            })
        
        results['total_checked'] = self._get_total_connections() * 2  # source + target
        return results
    
    def _validate_value_formats(self) -> Dict[str, Any]:
        """値の形式を検証"""
        results = {
            'total_checked': 0,
            'issues_found': 0,
            'issue_types': defaultdict(int),
            'value_patterns': defaultdict(int)
        }
        
        # 値を持つ接続を取得
        query = """
        SELECT source_object_name, source_value, target_object_name, target_value
        FROM object_connections
        WHERE source_value IS NOT NULL OR target_value IS NOT NULL
        LIMIT 10000
        """
        connections_with_values = self.db.execute_query(query)
        
        for conn in connections_with_values:
            # ソース値の検証
            if conn['source_value']:
                self._validate_single_value(
                    conn['source_value'], 
                    conn['source_object_name'],
                    results
                )
            
            # ターゲット値の検証
            if conn['target_value']:
                self._validate_single_value(
                    conn['target_value'],
                    conn['target_object_name'],
                    results
                )
        
        results['total_checked'] = len(connections_with_values)
        return results
    
    def _validate_single_value(self, value: str, object_name: str, results: Dict) -> None:
        """個別の値を検証"""
        # 値のパターンを分類
        if value.isdigit():
            results['value_patterns']['integer'] += 1
        elif re.match(r'^-?\d+\.?\d*$', value):
            results['value_patterns']['float'] += 1
        elif value in ['bang', 'set', 'clear', 'dump']:
            results['value_patterns']['standard_message'] += 1
        elif value.startswith('[') and value.endswith(']'):
            results['value_patterns']['list'] += 1
            # リスト形式の検証
            try:
                json.loads(value)
            except json.JSONDecodeError:
                results['issues_found'] += 1
                results['issue_types']['invalid_list_format'] += 1
        elif len(value) > 1000:
            results['issues_found'] += 1
            results['issue_types']['excessively_long_value'] += 1
            self.issues.append({
                'type': 'excessively_long_value',
                'object': object_name,
                'severity': 'warning',
                'description': f'値が異常に長いです（{len(value)}文字）'
            })
        else:
            results['value_patterns']['other'] += 1
    
    def _validate_hierarchy(self) -> Dict[str, Any]:
        """階層構造の整合性を検証"""
        results = {
            'total_checked': 0,
            'issues_found': 0,
            'issue_types': defaultdict(int),
            'max_depth': 0
        }
        
        # parent_contextの深さを分析
        query = """
        SELECT parent_context, 
               LENGTH(parent_context) - LENGTH(REPLACE(parent_context, ':', '')) as depth
        FROM object_connections
        WHERE parent_context IS NOT NULL AND parent_context != ''
        """
        hierarchy_data = self.db.execute_query(query)
        
        if hierarchy_data:
            depths = [row['depth'] for row in hierarchy_data]
            results['max_depth'] = max(depths) if depths else 0
            
            # 異常に深い階層をチェック（通常は4-5レベルまで）
            deep_hierarchies = [row for row in hierarchy_data if row['depth'] > 5]
            if deep_hierarchies:
                results['issues_found'] += len(deep_hierarchies)
                results['issue_types']['excessively_deep_hierarchy'] = len(deep_hierarchies)
                self.issues.append({
                    'type': 'excessively_deep_hierarchy',
                    'count': len(deep_hierarchies),
                    'severity': 'warning',
                    'description': f'{len(deep_hierarchies)}個の接続で異常に深い階層（>5レベル）が検出されました',
                    'examples': [row['parent_context'] for row in deep_hierarchies[:3]]
                })
        
        # gen~コンテキストの検証
        query = """
        SELECT COUNT(*) as count
        FROM object_connections
        WHERE parent_context LIKE '%gen~%'
        AND (source_object_name NOT IN ('Param', 'History', 'Data', 'Buffer', 'Sample', 'Poke', 
                                       'Peek', 'Dim', 'Channels', 'Delay', 'Latch', 'Change',
                                       'Delta', 'Scale', 'Clip', 'Wrap', 'Fold', 'tanh', 'dcblock')
             AND target_object_name NOT IN ('Param', 'History', 'Data', 'Buffer', 'Sample', 'Poke',
                                           'Peek', 'Dim', 'Channels', 'Delay', 'Latch', 'Change',
                                           'Delta', 'Scale', 'Clip', 'Wrap', 'Fold', 'tanh', 'dcblock'))
        """
        result = self.db.execute_query(query)
        invalid_gen_objects = result[0]['count'] if result else 0
        
        if invalid_gen_objects > 0:
            results['issues_found'] += invalid_gen_objects
            results['issue_types']['invalid_gen_object'] = invalid_gen_objects
            self.issues.append({
                'type': 'invalid_gen_object',
                'count': invalid_gen_objects,
                'severity': 'info',
                'description': f'{invalid_gen_objects}個の接続でgen~内に通常のMaxオブジェクトが検出されました'
            })
        
        results['total_checked'] = len(hierarchy_data) if hierarchy_data else 0
        return results
    
    def _validate_duplicates(self) -> Dict[str, Any]:
        """重複接続を検証"""
        results = {
            'total_checked': 0,
            'duplicates_found': 0,
            'duplicate_groups': []
        }
        
        # 完全に同一の接続を検索
        query = """
        SELECT patch_file, source_object_name, source_object_id, source_port,
               target_object_name, target_object_id, target_port,
               COUNT(*) as duplicate_count
        FROM object_connections
        GROUP BY patch_file, source_object_name, source_object_id, source_port,
                 target_object_name, target_object_id, target_port
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC
        LIMIT 10
        """
        duplicates = self.db.execute_query(query)
        
        if duplicates:
            total_duplicates = sum(row['duplicate_count'] - 1 for row in duplicates)
            results['duplicates_found'] = total_duplicates
            results['duplicate_groups'] = [
                {
                    'patch': row['patch_file'],
                    'connection': f"{row['source_object_name']}:{row['source_port']} -> {row['target_object_name']}:{row['target_port']}",
                    'count': row['duplicate_count']
                }
                for row in duplicates
            ]
            
            self.issues.append({
                'type': 'duplicate_connections',
                'count': total_duplicates,
                'severity': 'error',
                'description': f'{total_duplicates}個の重複接続が検出されました',
                'top_duplicates': results['duplicate_groups'][:5]
            })
        
        results['total_checked'] = self._get_total_connections()
        return results
    
    def _validate_gen_objects(self) -> Dict[str, Any]:
        """Gen~内部オブジェクトの検証"""
        results = {
            'total_gen_connections': 0,
            'valid_gen_objects': 0,
            'invalid_gen_objects': 0,
            'gen_object_types': defaultdict(int)
        }
        
        # Gen~コンテキスト内の接続を取得
        query = """
        SELECT source_object_name, target_object_name
        FROM object_connections
        WHERE parent_context LIKE '%gen~%'
        """
        gen_connections = self.db.execute_query(query)
        
        # Gen~で使用可能なオブジェクト
        valid_gen_objects = {
            'Param', 'History', 'Data', 'Buffer', 'Sample', 'Poke', 'Peek',
            'Dim', 'Channels', 'Delay', 'Latch', 'Change', 'Delta',
            'Scale', 'Clip', 'Wrap', 'Fold', 'tanh', 'dcblock',
            '+', '-', '*', '/', '%', 'pow', 'sqrt', 'abs', 'min', 'max',
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
            'noise', 'cycle', 'phasor', 'train', 'rate', 'in', 'out'
        }
        
        for conn in gen_connections:
            for obj_name in [conn['source_object_name'], conn['target_object_name']]:
                if obj_name:
                    base_name = obj_name.split()[0] if ' ' in obj_name else obj_name
                    results['gen_object_types'][base_name] += 1
                    
                    if base_name in valid_gen_objects:
                        results['valid_gen_objects'] += 1
                    else:
                        results['invalid_gen_objects'] += 1
        
        results['total_gen_connections'] = len(gen_connections)
        
        if results['invalid_gen_objects'] > 0:
            self.issues.append({
                'type': 'invalid_gen_objects',
                'count': results['invalid_gen_objects'],
                'severity': 'info',
                'description': f"Gen~内で{results['invalid_gen_objects']}個の非標準オブジェクトが使用されています",
                'note': 'これらは互換性の問題を引き起こす可能性があります'
            })
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """検証結果のレポートを生成"""
        report = []
        report.append("=" * 80)
        report.append("Max/MSPパッチ接続データ検証レポート")
        report.append("=" * 80)
        report.append(f"\n総接続数: {results['total_connections']:,}")
        
        # 検証結果のサマリー
        total_issues = len(results['issues'])
        report.append(f"\n検出された問題: {total_issues}件")
        
        if total_issues > 0:
            # 重要度別に分類
            severity_counts = defaultdict(int)
            for issue in results['issues']:
                severity_counts[issue['severity']] += 1
            
            report.append("\n重要度別内訳:")
            for severity in ['error', 'warning', 'info']:
                if severity in severity_counts:
                    report.append(f"  - {severity.upper()}: {severity_counts[severity]}件")
        
        # 各検証の詳細
        report.append("\n" + "=" * 80)
        report.append("検証結果詳細")
        report.append("=" * 80)
        
        for validation_name, validation_result in results['validation_results'].items():
            report.append(f"\n### {validation_name.replace('_', ' ').title()}")
            
            if isinstance(validation_result, dict):
                for key, value in validation_result.items():
                    if key != 'issue_types' and not key.endswith('_groups'):
                        report.append(f"  - {key}: {value}")
                
                if 'issue_types' in validation_result and validation_result['issue_types']:
                    report.append("  問題タイプ:")
                    for issue_type, count in validation_result['issue_types'].items():
                        report.append(f"    - {issue_type}: {count}件")
        
        # 問題の詳細
        if results['issues']:
            report.append("\n" + "=" * 80)
            report.append("問題の詳細")
            report.append("=" * 80)
            
            for issue in sorted(results['issues'], key=lambda x: ['error', 'warning', 'info'].index(x['severity'])):
                report.append(f"\n[{issue['severity'].upper()}] {issue['type']}")
                report.append(f"  {issue['description']}")
                
                if 'details' in issue and issue['details']:
                    report.append("  詳細:")
                    for detail in issue['details'][:3]:
                        report.append(f"    - {detail}")
                
                if 'examples' in issue and issue['examples']:
                    report.append("  例:")
                    for example in issue['examples'][:3]:
                        report.append(f"    - {example}")
        
        # レポートの出力
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\nレポートを保存しました: {output_path}")
        else:
            print(report_text)
        
        # JSON形式でも保存
        if output_path:
            json_path = Path(output_path).with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"JSON形式でも保存しました: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Max/MSPパッチ接続データの検証'
    )
    parser.add_argument(
        '--config',
        default='./scripts/db_settings.ini',
        help='データベース設定ファイルのパス'
    )
    parser.add_argument(
        '--output',
        help='レポート出力ファイルのパス（指定しない場合は標準出力）'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='JSON形式のみ出力'
    )
    
    args = parser.parse_args()
    
    # 検証を実行
    validator = DataValidator(args.config)
    results = validator.validate_all()
    
    # レポートを生成
    if args.json_only and args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"検証結果をJSON形式で保存しました: {args.output}")
    else:
        validator.generate_report(results, args.output)
    
    # 問題がある場合は終了コード1を返す
    if results['issues']:
        exit(1)


if __name__ == '__main__':
    main()