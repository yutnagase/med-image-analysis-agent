"""医療画像解析AIエージェント - ステップ1: 画像解析PoC.

Streamlit UIから医療画像をアップロードし、
Ollama経由でqwen2.5-vl:3bモデルによる画像解析を実行する。
"""

import base64
from pathlib import Path

import requests
import streamlit as st

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5vl:3b"
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "dicom"]

SYSTEM_PROMPT = (
    "You are a medical imaging analysis assistant. "
    "Describe the uploaded medical image in detail, "
    "including anatomical structures, any notable findings, "
    "and potential areas of concern. "
    "Respond in a structured, professional manner. "
    "Always respond in Japanese."
)


def check_ollama_health() -> bool:
    """Ollamaサーバーの稼働状態を確認する."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def analyze_image(image_bytes: bytes, prompt: str) -> str:
    """Ollama APIに画像を送信し、VLMによる解析結果を取得する.

    Args:
        image_bytes: アップロードされた画像のバイナリデータ.
        prompt: モデルに渡すユーザープロンプト.

    Returns:
        モデルが生成した解析テキスト.

    Raises:
        RuntimeError: API呼び出しに失敗した場合.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "images": [base64.b64encode(image_bytes).decode("utf-8")],
        "stream": False,
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=600,  # CPU駆動 + VLMのため長めのタイムアウト
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API error: {resp.status_code} - {resp.text}")

    return resp.json()["response"]


def main() -> None:
    """Streamlit UIのエントリーポイント."""
    st.set_page_config(
        page_title="Medical Image Analysis Agent",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 Medical Image Analysis Agent")
    st.caption(f"Model: `{MODEL_NAME}` | Runtime: Ollama (CPU)")

    # Ollamaヘルスチェック
    if not check_ollama_health():
        st.error(
            "⚠️ Ollamaサーバーに接続できません。\n\n"
            "```bash\nollama serve\n```\n\n"
            "を実行してからページをリロードしてください。"
        )
        st.stop()

    st.success("✅ Ollama接続確認済み")

    # 画像アップロード
    uploaded_file = st.file_uploader(
        "医療画像をアップロード",
        type=SUPPORTED_FORMATS,
        help="X線・CT・MRI等の画像ファイル (PNG/JPEG)",
    )

    if uploaded_file is None:
        st.info("👆 画像ファイルを選択してください。")
        return

    # プレビュー表示
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 アップロード画像")
        st.image(uploaded_file, use_container_width=True)

    # 解析実行
    with col2:
        st.subheader("📋 解析結果")

        if st.button("🔍 解析を実行", type="primary", use_container_width=True):
            image_bytes = uploaded_file.getvalue()

            try:
                with st.spinner("🧠 画像を解析中です（CPU駆動のため数十秒かかります）..."):
                    result = analyze_image(
                        image_bytes,
                        prompt="この医療画像を解析し、詳細な所見を日本語で報告してください。",
                    )
                st.markdown(result)

            except requests.ConnectionError:
                st.error("Ollamaサーバーとの接続が切断されました。")
            except requests.Timeout:
                st.error("解析がタイムアウトしました。システムリソースを確認してください。")
            except RuntimeError as e:
                st.error(f"解析エラー: {e}")


if __name__ == "__main__":
    main()
