#!/usr/bin/env python3
"""
Max/MSP AI パッチジェネレーター CLI
"""

import sys
import argparse
from max_patch_ai_engine import MaxPatchGenerator

def main():
    parser = argparse.ArgumentParser(
        description='🎵 Max/MSP AI パッチジェネレーター',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  python max_patch_cli.py "雨の音を表現したアンビエント"
  python max_patch_cli.py "aggressive distortion effect"
  python max_patch_cli.py --batch "reverb,delay,filter"
        '''
    )
    
    parser.add_argument('input', nargs='?', help='パッチの説明（日本語・英語対応）')
    parser.add_argument('--batch', help='カンマ区切りでバッチ生成')
    parser.add_argument('--interactive', '-i', action='store_true', help='対話モード')
    parser.add_argument('--no-cache', action='store_true', help='キャッシュを使用しない')
    
    args = parser.parse_args()
    
    generator = MaxPatchGenerator()
    
    if args.interactive:
        interactive_mode(generator, not args.no_cache)
    elif args.batch:
        batch_mode(generator, args.batch, not args.no_cache)
    elif args.input:
        single_mode(generator, args.input, not args.no_cache)
    else:
        parser.print_help()

def single_mode(generator, user_input, use_cache):
    """シングル生成モード"""
    print(f"🎵 Max/MSP AI パッチジェネレーター")
    print(f"📝 生成要求: {user_input}")
    
    result = generator.generate_patch(user_input, use_cache)
    
    print_result(result)

def batch_mode(generator, batch_input, use_cache):
    """バッチ生成モード"""
    requests = [req.strip() for req in batch_input.split(',')]
    
    print(f"🚀 バッチモード: {len(requests)}件")
    results = generator.batch_generate(requests)
    
    for result in results:
        print_result(result, compact=True)

def interactive_mode(generator, use_cache):
    """対話モード"""
    print("🎵 Max/MSP AI パッチジェネレーター - 対話モード")
    print("終了するには 'quit' または 'exit' を入力してください")
    
    while True:
        try:
            user_input = input("\\n🎨 パッチの説明を入力してください: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 ありがとうございました！")
                break
            
            if not user_input:
                continue
            
            result = generator.generate_patch(user_input, use_cache)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\\n👋 終了しました")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")

def print_result(result, compact=False):
    """結果表示"""
    intent = result.intent
    
    if compact:
        print(f"✅ {intent.concept} -> {result.file_path}")
    else:
        print(f"\\n🎨 生成結果:")
        print(f"  コンセプト: {intent.concept}")
        print(f"  カテゴリ: {intent.category}")
        print(f"  オブジェクト: {', '.join(intent.objects)}")
        print(f"  創造性スコア: {intent.creativity_score:.2f}")
        print(f"  生成時間: {result.generation_time:.2f}秒")
        print(f"  生成方法: {result.method_used}")
        print(f"  ファイル: {result.file_path}")

if __name__ == "__main__":
    main()