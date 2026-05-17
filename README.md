# 🏥 Medical Image Analysis Agent

医療用診断画像をVLM（Vision-Language Model）で解析し、ガイドライン検索・レポート生成まで自律的に行うAIエージェントのPoC。

## アーキテクチャ

```
[Streamlit UI]
      │
      ▼
┌─────────────────────────────────────────────┐
│         Agentic Workflow Pipeline            │
├─────────────────────────────────────────────┤
│ Step 1: VLM画像解析（所見抽出）             │
│    ↓                                        │
│ Step 2: Agentic RAG（ガイドライン知識検索） │
│    ↓                                        │
│ Step 3: 構造化臨床レポート自動生成          │
└─────────────────────────────────────────────┘
      │
      ▼
[Ollama API] → [qwen2.5vl:3b] (CPU駆動・単一モデルで全ステップ処理)
```

### Agentic RAG の仕組み

本システムは「Agentic RAG（自律型検索拡張生成）」パターンを採用しています:

1. **画像解析（Perception）**: VLMが医療画像から所見を抽出
2. **知識検索（Retrieval）**: LLMが所見を解釈し、模擬ガイドライン知識ベースから関連プロトコルを自律的に選択・抽出
3. **レポート生成（Generation）**: 所見とガイドラインを統合し、構造化された臨床レポートを生成

従来のRAGとの違いは、検索クエリの生成・結果の評価・情報統合をすべてLLMが自律的に判断する点にあります。

### リソース最適化設計

メモリ12GB・CPU駆動の制約下で、テキスト処理（RAG・レポート生成）もVLMモデル（qwen2.5vl:3b）に兼任させることで、追加モデルのロードを回避しメモリ使用量を最小化しています。

## 動作環境

| 項目 | 要件 |
|------|------|
| OS | Ubuntu (WSL2) |
| Python | 3.10+ |
| メモリ | 12GB (Swap: 4GB) |
| GPU | 不要（CPU駆動） |
| Ollama | インストール済み |

## セットアップ

### 1. Ollamaのインストールとモデル取得

```bash
# Ollamaインストール
curl -fsSL https://ollama.com/install.sh | sh

# VLMモデルのダウンロード
ollama pull qwen2.5vl:3b
```

> **Note:** `qwen2.5-vl:3b`（ハイフンあり）ではなく `qwen2.5vl:3b`（ハイフンなし）が正しいモデル名です。詳細は[トラブルシューティング](#トラブルシューティング)を参照。

### 2. Python仮想環境のセットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行方法

### Ollamaサーバー起動

```bash
ollama serve
```

> **Note:** `address already in use` エラーが出る場合、Ollamaは既にバックグラウンドで起動済みです。そのまま次に進んでください。

### Streamlitアプリ起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

## 使い方

1. 画面上で医療画像（PNG/JPEG）をアップロード
2. 「🔍 エージェント解析を実行」ボタンをクリック
3. 3ステップのパイプラインが順次実行される:
   - Step 1: VLMによる画像所見の抽出
   - Step 2: ガイドラインからの推奨アクション検索
   - Step 3: 医師向け臨床レポートの自動生成
4. タブ切替で「画像所見」「ガイドライン検索結果」「臨床レポート」を確認

## トラブルシューティング

開発中に遭遇した問題と解決策を記録しています。

### モデル名の不一致（`file does not exist`）

```
Error: pull model manifest: file does not exist
```

**原因:** Ollamaレジストリ上のモデル名は `qwen2.5vl:3b`（ハイフンなし）。`qwen2.5-vl:3b` は存在しないためマニフェスト取得に失敗する。

**解決:** ハイフンを除いた正しいモデル名を使用。
```bash
# NG
ollama pull qwen2.5-vl:3b

# OK
ollama pull qwen2.5vl:3b
```

### Ollamaサーバーの重複起動（`address already in use`）

```
Error: listen tcp 127.0.0.1:11434: bind: address already in use
```

**原因:** Ollamaはインストール時にsystemdサービスとして自動登録され、バックグラウンドで既に起動している。

**解決:** エラーではないため無視して問題なし。`ollama ps` でモデルの状態を確認可能。

### VLM画像解析時のGGMLアサーションエラー

```
GGML_ASSERT(a->ne[2] * 4 == b->ne[0]) failed
```

**原因:** 入力画像の解像度が大きすぎ、モデル内部のテンソル演算でサイズ不整合が発生。qwen2.5vl:3bのビジョンエンコーダが想定する入力サイズを超過していた。

**解決:** 画像をモデルに渡す前に512px以下にリサイズする前処理を追加。Pillowライブラリで `thumbnail()` を使用し、アスペクト比を維持したまま縮小。

```python
def resize_image(image_bytes: bytes, max_size: int = 512) -> bytes:
    img = Image.open(BytesIO(image_bytes))
    img.thumbnail((max_size, max_size))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()
```

### CPU駆動でのタイムアウト

**原因:** 12GB RAM + CPU駆動で3.8Bパラメータモデルを推論するため、初回はモデルロード（3.2GB）＋推論で5〜10分かかる場合がある。

**解決:**
- APIタイムアウトを600秒（10分）に設定
- 2回目以降はモデルがメモリに残るため高速化（ただし一定時間未使用で自動アンロードされる）
- `ollama ps` でモデルのロード状態を確認可能

## 開発ログ

| 段階 | 内容 | 課題と対応 |
|------|------|-----------|
| Step 1 | VLM画像解析PoC | モデル名の表記揺れ、CPU環境でのタイムアウト調整 |
| Step 2 | Agentic RAG + レポート生成 | 高解像度画像によるGGMLエラー → リサイズ前処理で解決 |
