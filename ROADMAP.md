# MANXO Development Roadmap

## 🎯 Project Goal

自然言語からMax/MSPパッチを生成するAIシステムの構築

## 📍 Current Status (2025年5月)

- ✅ 689,098接続パターンをPostgreSQLに保存済み
- ✅ 基本的なプロジェクト構造
- ❌ 実動作するコードはほぼゼロ

## 🚀 Development Phases

### Phase 0: Foundation (現在) - 2週間

**目標**: 基盤となるツールを動作させる

1. **データベース検証** [Issue #16]
   - [ ] `scripts/verify_database.py` - DB内容を確認
   - [ ] `scripts/export_sample_data.py` - サンプルデータ抽出
   - [ ] 実際のデータベース接続テスト

2. **基本的なデータ探索** [Issue #17]
   - [ ] `scripts/explore_patterns.py` - 頻出パターン分析
   - [ ] `scripts/visualize_connections.py` - 接続の可視化
   - [ ] よく使われるオブジェクトの統計

3. **シンプルなパッチ生成** [Issue #18]
   - [ ] `scripts/generate_basic_patch.py` - テンプレートベース生成
   - [ ] オシレーター、フィルター、DAC~の組み合わせ
   - [ ] JSONファイル出力と検証

### Phase 1: Pattern Learning - 1ヶ月

**目標**: データから基本パターンを学習

4. **グラフデータセット作成** [Issue #19]
   - [ ] `scripts/create_graph_dataset.py` - PyTorch Geometric形式
   - [ ] ノード特徴量設計（オブジェクトタイプ、位置、値）
   - [ ] エッジ特徴量設計（ポート番号、データタイプ）

5. **シンプルなGNN実装** [Issue #20]
   - [ ] 2-3層の基本的なGCN
   - [ ] 接続予測タスク（次に来るオブジェクトは？）
   - [ ] 精度60%を目標

6. **パターンマッチング** [Issue #21]
   - [ ] よくある接続パターンのテンプレート化
   - [ ] "cycle~ → *~ → dac~"のような頻出パターン
   - [ ] テンプレートからの生成

### Phase 2: Basic Generation - 2ヶ月

**目標**: 簡単な指示からパッチを生成

7. **ルールベース生成** [Issue #22]
   - [ ] "オシレーター" → cycle~
   - [ ] "フィルター" → biquad~
   - [ ] 基本的な音響効果の実装

8. **キーワードマッチング** [Issue #23]
   - [ ] 自然言語 → Max用語の辞書
   - [ ] "リバーブ" → freeverb~
   - [ ] 複合キーワードの処理

9. **構造的な生成** [Issue #24]
   - [ ] 入力 → 処理 → 出力の基本構造
   - [ ] オブジェクト間の適切な接続
   - [ ] 動作するパッチの保証

### Phase 3: AI Integration - 3ヶ月

**目標**: 本格的なAI機能の実装

10. **Neural Knowledge Base** [Issue #1 改訂]
    - [ ] 実際のデータでの実装
    - [ ] 類似パターン検索
    - [ ] 効率的なインデックス

11. **高度なGNN** [Issue #2 改訂]
    - [ ] GraphSAGE/GAT実装
    - [ ] 階層的グラフ表現
    - [ ] 90%以上の予測精度

12. **自然言語理解** [Issue #3 改訂]
    - [ ] 事前学習済みモデルの活用
    - [ ] Max/MSP用語の埋め込み
    - [ ] 意図理解と生成

## 📊 Success Metrics

### Phase 0 (2週間後)
- [ ] 10種類の基本パッチを生成可能
- [ ] データベースの内容を完全に理解
- [ ] 3つ以上の動作するスクリプト

### Phase 1 (1.5ヶ月後)
- [ ] 100種類のパターンを学習
- [ ] GNNで60%の予測精度
- [ ] グラフデータセット完成

### Phase 2 (3.5ヶ月後)
- [ ] 50個のキーワードに対応
- [ ] 生成パッチの80%が動作
- [ ] デモ可能な状態

### Phase 3 (6.5ヶ月後)
- [ ] 自然な日本語入力に対応
- [ ] 複雑なエフェクトチェーン生成
- [ ] 実用レベルの品質

## 🛠️ 必要なスキル

- **すぐ必要**: Python, PostgreSQL, JSON
- **Phase 1から**: PyTorch, NetworkX
- **Phase 2から**: Max/MSP知識
- **Phase 3から**: NLP, Transformer

## 💭 Reality Check

このプロジェクトは野心的です。しかし：
- Phase 0-1は既存技術で実現可能
- Phase 2でも実用的な成果が期待できる
- Phase 3は研究レベルの挑戦

**まずはPhase 0を2週間で完了させることから始めましょう。**