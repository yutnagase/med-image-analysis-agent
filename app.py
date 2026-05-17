"""医療画像解析AIエージェント - 自律型診断支援システム.

ワークフロー:
1. VLMによる画像解析（所見抽出）
2. Agentic RAG（ガイドライン知識検索）
3. 構造化臨床レポートの自動生成
"""

import base64
from datetime import datetime
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# --- 設定 ---
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5vl:3b"
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "dicom"]

# --- 模擬ガイドライン知識ベース（Mock RAG） ---
CLINICAL_GUIDELINES: dict[str, dict[str, str]] = {
    "normal": {
        "condition": "正常所見",
        "action": "特記すべき異常なし。定期健診スケジュールに従い経過観察を継続。",
        "urgency": "低",
    },
    "cardiomegaly": {
        "condition": "心拡大（Cardiomegaly）",
        "action": (
            "心胸郭比（CTR）の計測を実施。CTR > 50%の場合、"
            "心エコー検査を追加オーダーし、心不全・弁膜症・高血圧性心疾患の鑑別を行う。"
            "BNP/NT-proBNP測定を推奨。"
        ),
        "urgency": "中",
    },
    "pneumonia": {
        "condition": "肺炎（Pneumonia）",
        "action": (
            "浸潤影の範囲・分布を評価。市中肺炎の場合、A-DROPスコアで重症度判定。"
            "軽症: 外来で経口抗菌薬（アモキシシリン等）投与。"
            "中等症以上: 入院加療、血液培養2セット採取後に経験的抗菌薬投与。"
            "48-72時間後に胸部X線再検を推奨。"
        ),
        "urgency": "高",
    },
    "pleural_effusion": {
        "condition": "胸水（Pleural Effusion）",
        "action": (
            "胸水量の推定（少量/中等量/大量）。原因検索として心不全、感染症、"
            "悪性腫瘍を鑑別。中等量以上の場合、胸腔穿刺による検体採取を検討。"
            "Light's criteriaで滲出性/漏出性を判定。"
        ),
        "urgency": "中〜高",
    },
    "pneumothorax": {
        "condition": "気胸（Pneumothorax）",
        "action": (
            "虚脱率を評価。軽度（20%未満）: 安静経過観察、24時間後に再撮影。"
            "中等度以上: 胸腔ドレナージを実施。緊張性気胸の徴候がある場合は緊急脱気。"
        ),
        "urgency": "高〜緊急",
    },
    "fracture": {
        "condition": "骨折（Fracture）",
        "action": (
            "骨折部位・転位の有無を評価。肋骨骨折の場合、"
            "気胸・血胸の合併を除外。疼痛管理と呼吸リハビリテーションを開始。"
            "多発肋骨骨折ではフレイルチェストに注意。"
        ),
        "urgency": "中",
    },
}

DISCLAIMER = (
    "⚠️ **免責事項**: 本レポートはAIによる診断支援情報であり、"
    "最終的な臨床判断は担当医師が行ってください。"
    "本システムの出力は医療行為における確定診断を構成するものではありません。"
)


# --- Ollama API ---
def check_ollama_health() -> bool:
    """Ollamaサーバーの稼働状態を確認する."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def call_llm(prompt: str, system: str, images: list[str] | None = None) -> str:
    """Ollama APIを呼び出し、LLM/VLMの応答を取得する.

    Args:
        prompt: ユーザープロンプト.
        system: システムプロンプト.
        images: Base64エンコード済み画像リスト（VLM使用時）.

    Returns:
        モデルの応答テキスト.

    Raises:
        RuntimeError: API呼び出しに失敗した場合.
    """
    payload: dict = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    if images:
        payload["images"] = images

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=600,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API error: {resp.status_code} - {resp.text}")

    return resp.json()["response"]


# --- エージェント・ワークフロー ---
def resize_image(image_bytes: bytes, max_size: int = 512) -> bytes:
    """画像を指定サイズ以下にリサイズする.

    Args:
        image_bytes: 元画像のバイナリデータ.
        max_size: 長辺の最大ピクセル数.

    Returns:
        リサイズ後のJPEGバイナリデータ.
    """
    img = Image.open(BytesIO(image_bytes))
    img.thumbnail((max_size, max_size))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def step1_analyze_image(image_bytes: bytes) -> str:
    """ステップ1: VLMによる画像解析（所見抽出）."""
    resized = resize_image(image_bytes)
    return call_llm(
        prompt="この医療画像を解析し、詳細な所見を日本語で報告してください。",
        system=(
            "You are a medical imaging analysis assistant. "
            "Describe the uploaded medical image in detail, "
            "including anatomical structures, any notable findings, "
            "and potential areas of concern. "
            "Respond in a structured, professional manner. "
            "Always respond in Japanese."
        ),
        images=[base64.b64encode(resized).decode("utf-8")],
    )


def step2_search_guidelines(findings: str) -> str:
    """ステップ2: Agentic RAG - 所見に基づくガイドライン検索.

    LLMに所見を読み込ませ、知識ベースから適切なガイドラインを
    自律的に選択・抽出させる。
    """
    guidelines_text = "\n".join(
        f"- キー: {key} | 疾患: {g['condition']} | 対応: {g['action']} | 緊急度: {g['urgency']}"
        for key, g in CLINICAL_GUIDELINES.items()
    )

    prompt = (
        f"以下は画像解析で得られた所見です:\n\n{findings}\n\n"
        f"以下は利用可能な臨床ガイドラインです:\n\n{guidelines_text}\n\n"
        "上記の所見に最も関連するガイドラインを選択し、"
        "なぜそのガイドラインが適用されるのか理由を含めて日本語で回答してください。"
        "該当するガイドラインが複数ある場合はすべて挙げてください。"
    )

    return call_llm(
        prompt=prompt,
        system=(
            "You are a clinical decision support agent. "
            "Based on imaging findings, search and retrieve the most relevant "
            "clinical guidelines. Explain your reasoning. "
            "Always respond in Japanese."
        ),
    )


def step3_generate_report(findings: str, guideline_result: str) -> str:
    """ステップ3: 構造化臨床レポートの自動生成."""
    prompt = (
        f"以下の情報を基に、医師向けの構造化された診断支援レポートを生成してください。\n\n"
        f"【画像所見】\n{findings}\n\n"
        f"【ガイドライン検索結果】\n{guideline_result}\n\n"
        "以下のMarkdown形式で出力してください:\n"
        "## 画像所見サマリー\n（所見の要約）\n\n"
        "## 推奨アクション\n（ガイドラインに基づく具体的な対応策）\n\n"
        "## 緊急度\n（低/中/高/緊急）\n\n"
        "## 追加検査の提案\n（必要に応じて）\n"
    )

    return call_llm(
        prompt=prompt,
        system=(
            "You are a medical report generation assistant. "
            "Generate a structured clinical support report in Japanese. "
            "Be concise, professional, and actionable."
        ),
    )


# --- UI ---
def main() -> None:
    """Streamlit UIのエントリーポイント."""
    st.set_page_config(
        page_title="Medical Image Analysis Agent",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 Medical Image Analysis Agent")
    st.caption(f"Model: `{MODEL_NAME}` | Runtime: Ollama (CPU) | Agentic RAG Pipeline")

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

    # レイアウト: 左=画像、右=解析結果（タブ切替）
    col_image, col_result = st.columns([1, 2])

    with col_image:
        st.subheader("📷 アップロード画像")
        st.image(uploaded_file, use_container_width=True)

    with col_result:
        if st.button("🔍 エージェント解析を実行", type="primary", use_container_width=True):
            image_bytes = uploaded_file.getvalue()

            try:
                # ステップ1: 画像解析
                with st.spinner("🧠 Step 1/3: VLMによる画像解析中..."):
                    findings = step1_analyze_image(image_bytes)
                st.session_state["findings"] = findings

                # ステップ2: ガイドライン検索
                with st.spinner("📚 Step 2/3: ガイドライン検索中..."):
                    guideline_result = step2_search_guidelines(findings)
                st.session_state["guideline_result"] = guideline_result

                # ステップ3: レポート生成
                with st.spinner("📝 Step 3/3: 臨床レポート生成中..."):
                    report = step3_generate_report(findings, guideline_result)
                st.session_state["report"] = report

            except requests.ConnectionError:
                st.error("Ollamaサーバーとの接続が切断されました。")
            except requests.Timeout:
                st.error("解析がタイムアウトしました。システムリソースを確認してください。")
            except RuntimeError as e:
                st.error(f"解析エラー: {e}")

        # 結果表示（タブ切替）
        if "findings" in st.session_state:
            tab1, tab2, tab3 = st.tabs([
                "📋 画像所見",
                "📚 ガイドライン検索",
                "📄 臨床レポート",
            ])

            with tab1:
                st.markdown(st.session_state["findings"])

            with tab2:
                st.markdown(st.session_state["guideline_result"])

            with tab3:
                st.markdown(st.session_state["report"])
                st.divider()
                st.markdown(DISCLAIMER)
                st.caption(
                    f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Model: {MODEL_NAME}"
                )


if __name__ == "__main__":
    main()
