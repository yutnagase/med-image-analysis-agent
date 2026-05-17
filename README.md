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
│ Step 3: 類似症例検索（Case Retrieval）      │
│    ↓                                        │
│ Step 4: 自己評価ループ（Self-RAG）           │
│    │  └→ INSUFFICIENT → 再検索（最大1回）  │
│    ↓                                        │
│ Step 5: 構造化臨床レポート自動生成 + PDF    │
└─────────────────────────────────────────────┘
      │
      ▼
[Ollama API] → [qwen2.5vl:3b] (CPU駆動・単一モデルで全ステップ処理)
```

### Agentic RAG の仕組み

本システムは「Agentic RAG（自律型検索拡張生成）」パターンを採用しています:

1. **画像解析（Perception）**: VLMが医療画像から所見を抽出
2. **知識検索（Retrieval）**: LLMが所見を解釈し、模擬ガイドライン知識ベースから関連プロトコルを自律的に選択・抽出
3. **症例検索（Case Matching）**: 過去の確定診断データベースから類似症例を自律的に検索し、治療方針の参考情報を提供
4. **自己評価（Self-Evaluation）**: 検索結果の十分性をLLMが自律的に判定。不十分なら検索条件を広げて自動再検索
5. **レポート生成（Generation）**: 所見・ガイドライン・類似症例を統合し、構造化された臨床レポートを生成

従来のRAGとの違いは、検索クエリの生成・結果の評価・情報統合をすべてLLMが自律的に判断する点にあります。

### Self-RAG（自己評価ループ）

本システムは Corrective RAG パターンを採用し、検索結果の品質をエージェント自身が検証します:

```
検索完了 → 自己評価エージェントが十分性を判定
              │
              ├→ SUFFICIENT → レポート生成へ
              │
              └→ INSUFFICIENT → 検索条件を拡大して再検索（最大1回）
                                    ↓
                              レポート生成へ
```

これにより、固定パイプラインではなく「状況に応じて処理を分岐する自律的な意思決定」が実現されています。

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

### 3. 日本語フォントのインストール（PDF出力用）

```bash
sudo apt install -y fonts-noto-cjk
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
3. 4ステップ + 自己評価のパイプラインが順次実行される:
   - Step 1: VLMによる画像所見の抽出
   - Step 2: ガイドラインからの推奨アクション検索
   - Step 3: 過去の類似症例の検索・抽出
   - Step 4: エージェントによる自己評価（不十分なら再検索）
   - Step 5: 医師向け臨床レポートの自動生成
4. タブ切替で「画像所見」「ガイドライン検索結果」「類似症例」「臨床レポート」を確認
5. 「📥 PDFレポートをダウンロード」ボタンでレポートをPDF保存

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

## 既知の制約

### 軽量モデル（3B）によるハルシネーション

本システムで使用している `qwen2.5vl:3b` は汎用VLMであり、医療画像に特化した訓練を受けていません。そのため以下の挙動が確認されています:

- **撮影範囲外の臓器への言及**: 胸部X線で腹部臓器（肝臓・脾臓・胃など）が写らないのは正常だが、モデルが「確認できない＝異常の可能性」と誤って推論するケースがある
- **根拠のない所見の生成**: 小型モデルは「何か報告しなければ」という傾向が強く、画像から読み取れない情報を生成（ハルシネーション）する場合がある

**本PoCの位置づけ:**
本システムはエージェント・アーキテクチャの技術実証を目的としており、モデルの診断精度自体は検証対象外です。プロダクション環境では、医療画像に特化したファインチューニング済みモデル（例: RadFM, BiomedCLIP等）への置き換えや、Retrieval-Augmented Generationによる事実性の担保が必要になります。

### PDF生成時の水平スペースエラー

```
Not enough horizontal space to render a single character
```

**原因:** fpdf2で `cell(w=0)` を使用した後に `multi_cell(w=0)` を呼ぶと、カーソルのX位置がページ右端に残り、残り水平スペースが0と計算される。また、デフォルトの両端揃え（justify）により、`>` のような短い文字が行末に押し出される表示崩れも発生。

**解決:**
- `content_width = pdf.w - pdf.l_margin - pdf.r_margin` で有効描画幅を事前計算し、明示的に指定
- 各 `multi_cell` の前に `set_x(l_margin)` でカーソル位置をリセット
- 全テキストに `align="L"`（左揃え）を指定し、両端揃えによる文字飛びを防止
- `pdf.output()` が `bytearray` を返すため `bytes()` で変換してStreamlitに渡す

## 開発ログ

| 段階 | 内容 | 課題と対応 |
|------|------|-----------|
| Step 1 | VLM画像解析PoC | モデル名の表記揺れ、CPU環境でのタイムアウト調整 |
| Step 2 | Agentic RAG + レポート生成 | 高解像度画像によるGGMLエラー → リサイズ前処理で解決 |
| Step 3 | 類似症例検索 + PDFエクスポート | 日本語PDF出力のためNoto CJKフォント統合、fpdf2のカーソル位置バグ回避 |
