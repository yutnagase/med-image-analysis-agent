"""医療画像解析AIエージェント - 自律型診断支援システム.

ワークフロー:
0. モダリティ・ルーター（画像種別の自律判定 + ガードレール）
1. VLMによる画像解析（動的プロンプト注入による精密所見抽出）
2. Agentic RAG（ガイドライン知識検索 + 類似症例検索）
3. Self-RAG: 検索結果の自己評価 → 不十分なら再検索（最大1回）
4. 構造化臨床レポートの自動生成 + PDFエクスポート
"""

import base64
from datetime import datetime
from io import BytesIO

import requests
import streamlit as st
from fpdf import FPDF
from PIL import Image

# --- 設定 ---
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5vl:3b"
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "dicom"]

# --- モダリティ別読影チェックリスト（Dynamic Prompting用） ---
MODALITY_CHECKLIST: dict[str, dict[str, str]] = {
    "CHEST_XRAY": {
        "label": "胸部X線",
        "checklist": (
            "肺野外側の血管影の有無による気胸の有無、"
            "心陰影の拡大（心胸郭比CTR）、"
            "浸潤影や結節影の有無を重点的に確認せよ"
        ),
    },
    "BRAIN_MRI": {
        "label": "脳腫瘍MRI",
        "checklist": (
            "腫瘍の有無、周囲の浮腫（むくみ）の広がり、"
            "正中線構造のズレ（Midline Shift）の有無を重点的に確認せよ"
        ),
    },
}

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

# --- 模擬症例データベース（Mock Case DB） ---
CASE_DATABASE: list[dict[str, str]] = [
    {
        "case_id": "CXR-2024-001",
        "age": "67歳",
        "sex": "男性",
        "diagnosis": "右下肺野肺炎",
        "findings": "右下肺野にair bronchogramを伴う浸潤影",
        "treatment": "アモキシシリン/クラブラン酸 経口投与、7日間",
        "outcome": "72時間後の再検で改善確認、外来フォロー継続",
    },
    {
        "case_id": "CXR-2024-002",
        "age": "72歳",
        "sex": "女性",
        "diagnosis": "心拡大（高血圧性心疾患）",
        "findings": "CTR 58%、肺うっ血軽度、Kerley B lines陽性",
        "treatment": "利尿薬追加、降圧薬増量、心エコー精査",
        "outcome": "2週間後の再検でCTR改善（52%）、心エコーでEF 45%",
    },
    {
        "case_id": "CXR-2024-003",
        "age": "45歳",
        "sex": "男性",
        "diagnosis": "左自然気胸",
        "findings": "左肺虚脱率約30%、縦隔偏位なし",
        "treatment": "胸腔ドレナージ施行、持続吸引",
        "outcome": "5日後に肺拡張確認、ドレーン抜去",
    },
    {
        "case_id": "CXR-2024-004",
        "age": "58歳",
        "sex": "女性",
        "diagnosis": "右胸水（悪性胸膜中皮腫）",
        "findings": "右肋骨横隔膜角鈍化、中等量胸水貯留",
        "treatment": "胸腔穿刺で細胞診提出、胸膜生検追加",
        "outcome": "細胞診Class V、腫瘍内科へ紹介",
    },
    {
        "case_id": "CXR-2024-005",
        "age": "81歳",
        "sex": "男性",
        "diagnosis": "両側肺炎（誤嚥性）",
        "findings": "両側下肺野に斑状浸潤影、右優位",
        "treatment": "入院加療、スルバクタム/アンピシリン静注、嚥下リハ開始",
        "outcome": "7日後に改善傾向、14日後退院、嚥下機能評価継続",
    },
]

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


def step0_classify_modality(image_bytes: bytes) -> str:
    """ステップ0: 画像モダリティの自律判定（モダリティ・ルーター）.

    VLMが画像を分類し、CHEST_XRAY / BRAIN_MRI / UNKNOWN のいずれかを返す。
    """
    resized = resize_image(image_bytes)
    result = call_llm(
        prompt=(
            "この画像を分類してください。以下の3つのうち該当するものを1つだけ出力してください。\n"
            "- CHEST_XRAY（胸部X線画像の場合）\n"
            "- BRAIN_MRI（脳のMRI画像の場合）\n"
            "- UNKNOWN（上記以外、または医療画像でない場合）\n\n"
            "回答は CHEST_XRAY, BRAIN_MRI, UNKNOWN のいずれか1単語のみ出力してください。"
        ),
        system=(
            "You are a medical image modality classifier. "
            "Classify the image into exactly one category. "
            "Respond with ONLY one word: CHEST_XRAY, BRAIN_MRI, or UNKNOWN."
        ),
        images=[base64.b64encode(resized).decode("utf-8")],
    )
    for modality in ("CHEST_XRAY", "BRAIN_MRI", "UNKNOWN"):
        if modality in result.upper():
            return modality
    return "UNKNOWN"


def step1_analyze_image(image_bytes: bytes, modality: str) -> str:
    """ステップ1: VLMによる画像解析（動的プロンプト注入付き）."""
    resized = resize_image(image_bytes)
    checklist_info = MODALITY_CHECKLIST.get(modality, {})
    checklist_injection = (
        f"\n\n【読影重点チェックリスト】以下の点を特に注意して確認すること: "
        f"{checklist_info['checklist']}"
        if checklist_info
        else ""
    )
    return call_llm(
        prompt="この医療画像を解析し、詳細な所見を日本語で報告してください。",
        system=(
            "You are a medical imaging analysis assistant. "
            "Describe the uploaded medical image in detail, "
            "including anatomical structures, any notable findings, "
            "and potential areas of concern. "
            "Respond in a structured, professional manner. "
            "Always respond in Japanese."
            f"{checklist_injection}"
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


def step2b_search_similar_cases(findings: str) -> str:
    """ステップ2b: 類似症例の自律検索.

    LLMが所見を解釈し、症例DBから関連する過去症例を選択・要約する。
    """
    cases_text = "\n".join(
        f"- 症例ID: {c['case_id']} | 年齢: {c['age']} | 性別: {c['sex']} | "
        f"確定診断: {c['diagnosis']} | 所見: {c['findings']} | "
        f"治療: {c['treatment']} | 転帰: {c['outcome']}"
        for c in CASE_DATABASE
    )

    prompt = (
        f"以下は今回の画像解析で得られた所見です:\n\n{findings}\n\n"
        f"以下は過去の症例データベースです:\n\n{cases_text}\n\n"
        "今回の所見に最も類似する過去症例を1〜2件選択し、"
        "なぜ類似と判断したか、今回の診療にどう参考になるかを日本語で説明してください。"
    )

    return call_llm(
        prompt=prompt,
        system=(
            "You are a clinical case retrieval agent. "
            "Find the most similar past cases and explain relevance. "
            "Always respond in Japanese."
        ),
    )


def evaluate_search_result(
    findings: str, guideline_result: str, similar_cases: str
) -> bool:
    """検索結果の十分性を自己評価する（Self-RAG / Corrective RAG）.

    LLMに品質監査エージェントとして、検索結果が診断支援レポート作成に
    十分な情報を含んでいるかを判定させる。

    Args:
        findings: 画像所見テキスト.
        guideline_result: ガイドライン検索結果.
        similar_cases: 類似症例検索結果.

    Returns:
        True: 十分（SUFFICIENT）、False: 不十分（INSUFFICIENT）.
    """
    prompt = (
        f"以下の情報を検証してください。\n\n"
        f"【画像所見】\n{findings}\n\n"
        f"【検索されたガイドライン】\n{guideline_result}\n\n"
        f"【検索された類似症例】\n{similar_cases}\n\n"
        "上記の[画像所見]に対して、検索された[ガイドライン]と[類似症例]の内容が、"
        "診断支援レポートを作成するために十分な情報を含んでいるか検証してください。\n"
        "回答は必ず 'SUFFICIENT' または 'INSUFFICIENT' のいずれか1単語のみで出力してください。"
    )

    result = call_llm(
        prompt=prompt,
        system=(
            "You are an objective medical quality audit agent. "
            "Evaluate whether the retrieved information is sufficient "
            "to generate a clinical support report for the given findings. "
            "Respond with ONLY one word: 'SUFFICIENT' or 'INSUFFICIENT'."
        ),
    )

    return "SUFFICIENT" in result.upper()


def step2_retry_search(findings: str) -> tuple[str, str]:
    """検索条件を広げて再検索する（リトライ用）.

    初回検索で情報不足と判断された場合に、より広範な条件で
    ガイドラインと症例を再検索する。

    Args:
        findings: 画像所見テキスト.

    Returns:
        (ガイドライン検索結果, 類似症例検索結果) のタプル.
    """
    guidelines_text = "\n".join(
        f"- キー: {key} | 疾患: {g['condition']} | 対応: {g['action']} | 緊急度: {g['urgency']}"
        for key, g in CLINICAL_GUIDELINES.items()
    )

    guideline_result = call_llm(
        prompt=(
            f"以下は画像解析で得られた所見です:\n\n{findings}\n\n"
            f"以下は利用可能な臨床ガイドラインです:\n\n{guidelines_text}\n\n"
            "前回の検索では情報が不十分と判断されました。"
            "今回は所見に直接関連するものだけでなく、鑑別診断として考慮すべき"
            "ガイドラインも含めて、より広範に選択してください。"
            "各ガイドラインの適用理由を詳しく日本語で説明してください。"
        ),
        system=(
            "You are a clinical decision support agent performing a broader search. "
            "Include differential diagnoses and related conditions. "
            "Always respond in Japanese."
        ),
    )

    cases_text = "\n".join(
        f"- 症例ID: {c['case_id']} | 年齢: {c['age']} | 性別: {c['sex']} | "
        f"確定診断: {c['diagnosis']} | 所見: {c['findings']} | "
        f"治療: {c['treatment']} | 転帰: {c['outcome']}"
        for c in CASE_DATABASE
    )

    similar_cases = call_llm(
        prompt=(
            f"以下は今回の画像解析で得られた所見です:\n\n{findings}\n\n"
            f"以下は過去の症例データベースです:\n\n{cases_text}\n\n"
            "前回の検索では情報が不十分と判断されました。"
            "今回は類似度が高い症例だけでなく、鑑別診断の参考になる症例も含めて"
            "2〜3件選択し、それぞれの参考価値を日本語で詳しく説明してください。"
        ),
        system=(
            "You are a clinical case retrieval agent performing a broader search. "
            "Include cases useful for differential diagnosis. "
            "Always respond in Japanese."
        ),
    )

    return guideline_result, similar_cases


def step3_generate_report(
    findings: str, guideline_result: str, similar_cases: str
) -> str:
    """ステップ3: 構造化臨床レポートの自動生成."""
    prompt = (
        f"以下の情報を基に、医師向けの構造化された診断支援レポートを生成してください。\n\n"
        f"【画像所見】\n{findings}\n\n"
        f"【ガイドライン検索結果】\n{guideline_result}\n\n"
        f"【類似症例】\n{similar_cases}\n\n"
        "以下のMarkdown形式で出力してください:\n"
        "## 画像所見サマリー\n（所見の要約）\n\n"
        "## 推奨アクション\n（ガイドラインに基づく具体的な対応策）\n\n"
        "## 類似症例リファレンス\n（過去症例との比較と参考情報）\n\n"
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


# --- PDF生成 ---
def _strip_markdown(text: str) -> str:
    """Markdown記法をプレーンテキストに変換する.

    見出し行は「■」プレフィクスでマークし、PDF生成時にスタイル切替の目印とする。
    """
    import re
    text = re.sub(r"^##\s*(.+)$", r"■\1", text, flags=re.MULTILINE)
    text = re.sub(r"^#+\s*(.+)$", r"■\1", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^[-*]\s+", "・", text, flags=re.MULTILINE)
    return text


def generate_pdf(report: str, findings: str, guideline_result: str) -> bytes:
    """臨床レポートをPDFとして生成する.

    Args:
        report: 生成済みの臨床レポートテキスト.
        findings: 画像所見テキスト.
        guideline_result: ガイドライン検索結果テキスト.

    Returns:
        PDF文書のバイナリデータ.
    """
    pdf = FPDF()
    pdf.add_page()

    # 日本語フォント設定
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    pdf.add_font("NotoSansCJK", "", font_path)
    pdf.set_font("NotoSansCJK", size=10)

    # 有効描画幅を事前計算
    content_width = pdf.w - pdf.l_margin - pdf.r_margin

    # ヘッダー
    pdf.set_font("NotoSansCJK", size=16)
    pdf.cell(content_width, 12, "診断支援レポート", align="C")
    pdf.ln(14)
    pdf.set_font("NotoSansCJK", size=8)
    pdf.cell(
        content_width, 6,
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: {MODEL_NAME}",
        align="C",
    )
    pdf.ln(12)

    # 本文（Markdown除去済み）
    clean_report = _strip_markdown(report)
    pdf.set_font("NotoSansCJK", size=10)
    for line in clean_report.split("\n"):
        if line.strip() == "":
            pdf.ln(3)
        elif line.startswith("■"):
            pdf.ln(4)
            pdf.set_font("NotoSansCJK", size=12)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(content_width, 8, line.replace("■", ""), align="L")
            pdf.set_font("NotoSansCJK", size=10)
        else:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(content_width, 6, line, align="L")

    # 免責事項
    pdf.ln(10)
    pdf.set_font("NotoSansCJK", size=8)
    disclaimer_text = (
        "【免責事項】本レポートはAIによる診断支援情報であり、"
        "最終的な臨床判断は担当医師が行ってください。"
        "本システムの出力は医療行為における確定診断を構成するものではありません。"
    )
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(content_width, 5, disclaimer_text)

    return bytes(pdf.output())


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
                # ステップ0: モダリティ判定（ルーター）
                with st.spinner("🏷️ Step 0/5: 画像モダリティを自律判定中..."):
                    modality = step0_classify_modality(image_bytes)
                st.session_state["modality"] = modality

                # ガードレール: UNKNOWN なら即時停止
                if modality == "UNKNOWN":
                    st.error(
                        "🚫 医療用診断画像として認識できませんでした。"
                        "有効な画像をアップロードしてください。"
                    )
                    st.stop()

                # 判定結果の表示
                modality_info = MODALITY_CHECKLIST[modality]
                st.info(
                    f"🤖 エージェント判定: 画像を **{modality_info['label']}** "
                    f"({modality}) と判定し、専用の読影プロトコルを適用しました。"
                )

                # ステップ1: 動的プロンプト注入付き画像解析
                with st.spinner("🧠 Step 1/5: VLMによる画像解析中（専用チェックリスト適用）..."):
                    findings = step1_analyze_image(image_bytes, modality)
                st.session_state["findings"] = findings

                # ステップ2: ガイドライン検索
                with st.spinner("📚 Step 2/5: ガイドライン検索中..."):
                    guideline_result = step2_search_guidelines(findings)
                st.session_state["guideline_result"] = guideline_result

                # ステップ2b: 類似症例検索
                with st.spinner("🔎 Step 3/5: 類似症例検索中..."):
                    similar_cases = step2b_search_similar_cases(findings)
                st.session_state["similar_cases"] = similar_cases

                # ステップ2c: 自己評価ループ（Self-RAG）
                with st.spinner("🤖 Step 4/5: エージェントによる自己評価中..."):
                    is_sufficient = evaluate_search_result(
                        findings, guideline_result, similar_cases
                    )

                if is_sufficient:
                    st.success(
                        "✅ 自己評価: 十分な情報が収集されました。レポートを生成します。"
                    )
                else:
                    st.warning(
                        "⚠️ 自己評価: 情報不足と判断。検索条件を広げて再検索します..."
                    )
                    with st.spinner("🔄 再検索中（より広範な条件で実行）..."):
                        guideline_result, similar_cases = step2_retry_search(findings)
                    st.session_state["guideline_result"] = guideline_result
                    st.session_state["similar_cases"] = similar_cases
                    st.success("✅ 再検索完了。レポートを生成します。")

                st.session_state["self_eval_result"] = (
                    "SUFFICIENT" if is_sufficient else "INSUFFICIENT → 再検索実行"
                )

                # ステップ3: レポート生成
                with st.spinner("📝 Step 5/5: 臨床レポート生成中..."):
                    report = step3_generate_report(
                        findings, guideline_result, similar_cases
                    )
                st.session_state["report"] = report

            except requests.ConnectionError:
                st.error("Ollamaサーバーとの接続が切断されました。")
            except requests.Timeout:
                st.error("解析がタイムアウトしました。システムリソースを確認してください。")
            except RuntimeError as e:
                st.error(f"解析エラー: {e}")

        # 結果表示（タブ切替）
        if "findings" in st.session_state:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 画像所見",
                "📚 ガイドライン検索",
                "🔎 類似症例",
                "📄 臨床レポート",
            ])

            with tab1:
                st.markdown(st.session_state["findings"])

            with tab2:
                st.markdown(st.session_state["guideline_result"])

            with tab3:
                st.markdown(st.session_state["similar_cases"])

            with tab4:
                st.markdown(st.session_state["report"])
                st.divider()
                st.markdown(DISCLAIMER)
                st.caption(
                    f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Model: {MODEL_NAME}"
                )

                # PDFエクスポート
                try:
                    pdf_bytes = generate_pdf(
                        st.session_state["report"],
                        st.session_state["findings"],
                        st.session_state["guideline_result"],
                    )
                    st.download_button(
                        label="📥 PDFレポートをダウンロード",
                        data=pdf_bytes,
                        file_name=f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"PDF生成に失敗しました: {e}")


if __name__ == "__main__":
    main()
