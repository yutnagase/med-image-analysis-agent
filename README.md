# 🏥 Medical Image Analysis Agent

医療用診断画像をVLM（Vision-Language Model）で解析する自律型AIエージェントのPoC。

## アーキテクチャ

```
[Streamlit UI] → [Ollama API] → [qwen2.5-vl:3b] → 解析結果表示
```

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
2. 「🔍 解析を実行」ボタンをクリック
3. CPU駆動のため数十秒待機後、解析結果が表示される
