"""医療画像解析AIエージェント - 自律型診断支援システム.

ワークフロー:
0. モダリティ・ルーター（画像種別の自律判定 + ガードレール）
1. VLMによる画像解析（動的プロンプト注入による精密所見抽出）
2. Agentic RAG（ガイドライン知識検索 + 類似症例検索）
3. Self-RAG: 検索結果の自己評価 → 不十分なら再検索（最大1回）
4. 構造化臨床レポートの自動生成 + PDFエクスポート
"""

import base64
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from fpdf import FPDF
from PIL import Image

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- 設定 ---
OLLAMA_BASE_URL = "http://localhost:11434"
VLM_MODEL = "qwen2.5vl:7b"
TEXT_MODEL = "qwen3.5:4b"
KEEP_ALIVE = "30m"
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "dicom"]

# --- VLMモデル選択肢 ---
VLM_MODEL_OPTIONS = [
    {
        "id": "qwen2.5vl:7b",
        "label": "Qwen2.5-VL 7B ⭐ 推奨",
        "size": "6.0 GB",
        "ram": "8GB以上",
        "desc": "汎用VLM。安定動作・バランス良好（現在の標準モデル）",
    },
    {
        "id": "rohithbojja/llava-med-v1.6",
        "label": "LLaVA-Med v1.6（医療特化）",
        "size": "4.7 GB",
        "ram": "8GB以上",
        "desc": "医療画像データセットでFine-tuning済み ⚠️ Ollamaレジストリから削除される場合あり",
    },
    {
        "id": "minicpm-v:8b",
        "label": "MiniCPM-V 8B（軽量高性能）",
        "size": "5.5 GB",
        "ram": "8GB以上",
        "desc": "清華大学OpenBMB開発。軽量ながら高いVLM性能",
    },
    {
        "id": "llava:13b",
        "label": "LLaVA 13B（高精度汎用）",
        "size": "8.0 GB",
        "ram": "16GB以上",
        "desc": "7Bより高精度な汎用VLM。中スペック環境向け",
    },
    {
        "id": "llava:34b",
        "label": "LLaVA 34B（最高精度）",
        "size": "19.7 GB",
        "ram": "32GB以上",
        "desc": "最高精度。大容量RAM環境向け（M4 Pro 40GB等）",
    },
]

# --- 知識ベースのロード ---
def load_knowledge_base():
    """medical_documentsフォルダから全知識をロード"""
    base_dir = Path("./medical_documents")
    
    # 1. MODALITY_CHECKLIST
    modality_checklist = {}
    checklist_dir = base_dir / "modality_checklists"
    if checklist_dir.exists():
        for file in checklist_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    modality_checklist.update(data)
                logger.info(f"✅ チェックリスト読み込み成功: {file.name}")
            except Exception as e:
                logger.warning(f"モダリティチェックリスト読み込み失敗 {file}: {e}")
    
    # 2. CLINICAL_GUIDELINES
    clinical_guidelines = {}
    guidelines_dir = base_dir / "guidelines"
    if guidelines_dir.exists():
        for file in guidelines_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    key = file.stem
                    clinical_guidelines[key] = data
                logger.info(f"✅ ガイドライン読み込み成功: {file.name}")
            except Exception as e:
                logger.warning(f"ガイドライン読み込み失敗 {file}: {e}")
    
    # 3. CASE_DATABASE（新規追加）
    case_database = []
    cases_dir = base_dir / "cases"
    if cases_dir.exists():
        for file in cases_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    case_database.append(data)
                logger.info(f"✅ 症例データ読み込み成功: {file.name}")
            except Exception as e:
                logger.warning(f"症例データ読み込み失敗 {file}: {e}")
    
    # フォールバック
    if not modality_checklist:
        logger.warning("modality_checklists フォルダが見つからないか空のため、ハードコードを使用します。")
        modality_checklist = { ... }  # 既存の内容を維持

    if not clinical_guidelines:
        logger.warning("guidelines フォルダが見つからないか空のため、ハードコードを使用します。")
        clinical_guidelines = { ... }  # 既存の内容を維持

    if not case_database:
        logger.warning("cases フォルダが見つからないか空のため、ハードコードを使用します。")
        case_database = [
            {
                "case_id": "CXR-2024-001",
                "age": "67歳",
                "sex": "男性",
                "diagnosis": "右下肺野肺炎",
                "findings": "右下肺野にair bronchogramを伴う浸潤影",
                "treatment": "アモキシシリン/クラブラン酸 経口投与、7日間",
                "outcome": "72時間後の再検で改善確認",
            },
            {
                "case_id": "CXR-2024-002",
                "age": "72歳",
                "sex": "女性",
                "diagnosis": "心拡大（高血圧性心疾患）",
                "findings": "CTR 58%、肺うっ血軽度",
                "treatment": "利尿薬追加、降圧薬増量",
                "outcome": "2週間後の再検で改善",
            }
        ]
    
    return modality_checklist, clinical_guidelines, case_database

# ロード実行
MODALITY_CHECKLIST, CLINICAL_GUIDELINES, CASE_DATABASE = load_knowledge_base()

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


def warmup_model(model: str) -> bool:
    """モデルを事前ロードしてウォームアップする.

    空のプロンプトでモデルをメモリにロードさせ、
    初回推論時の不安定な応答を防止する。

    Returns:
        True: ウォームアップ成功、False: 失敗.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": KEEP_ALIVE},
            timeout=300,
        )
        if resp.status_code == 200:
            logger.info("[Warmup] Model '%s' loaded successfully.", model)
            return True
        logger.error("[Warmup] Failed to load '%s': %d", model, resp.status_code)
        return False
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.error("[Warmup] Error loading '%s': %s", model, e)
        return False


def call_llm(
    prompt: str, 
    system: str, 
    images: list[str] | None = None, 
    model: str | None = None
) -> str:
    """Ollama APIを呼び出し、LLM/VLMの応答を取得する.

    Args:
        prompt: ユーザープロンプト.
        system: システムプロンプト.
        images: Base64エンコード済み画像リスト（VLM使用時）.
        model: 使用モデル名。未指定時は画像有無で自動選択.

    Returns:
        モデルの応答テキスト.

    Raises:
        RuntimeError: API呼び出しに失敗した場合.
    """
    selected_model = model or (VLM_MODEL if images else TEXT_MODEL)
    payload = {
        "model": selected_model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
    }
    if images:
        payload["images"] = images

    logger.info("[LLM Request] model=%s, has_images=%s", selected_model, bool(images))
    logger.debug("[LLM Request] prompt=%s", prompt[:200])

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=1200,
    )

    if resp.status_code != 200:
        logger.error("[LLM Error] status=%d, body=%s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Ollama API error: {resp.status_code} - {resp.text}")

    response_text = resp.json()["response"]
    logger.info("[LLM Response] model=%s, length=%d", selected_model, len(response_text))
    logger.debug("[LLM Response] content=%s", response_text[:500])
    return response_text


# --- エージェント機能 ---
def resize_image(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """画像を指定サイズ以下にリサイズする.

    Args:
        image_bytes: 元画像のバイナリデータ.
        max_size: 長辺の最大ピクセル数.

    Returns:
        リサイズ後のJPEGバイナリデータ.
    """
    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def _single_classify(image_b64: str, vlm_model: str = VLM_MODEL) -> str:
    """単一回のモダリティ分類を実行する."""
    result = call_llm(
        prompt="この画像を分類してください。CHEST_XRAY, BRAIN_MRI, UNKNOWN のいずれか1単語のみ出力。",
        system="You are a medical image modality classifier. Respond with ONLY one word.",
        images=[image_b64],
        model=vlm_model,
    )
    for modality in ("CHEST_XRAY", "BRAIN_MRI", "UNKNOWN"):
        if modality in result.upper():
            return modality
    return "UNKNOWN"


def step0_classify_modality(image_bytes: bytes, vlm_model: str = VLM_MODEL) -> str:
    """ステップ0: 画像モダリティの自律判定（多数決方式）.

    VLMの分類精度が不安定なため3回試行し、多数決で最終判定する。
    """
    resized = resize_image(image_bytes)
    image_b64 = base64.b64encode(resized).decode("utf-8")
    logger.info(
        "[Step0] input_size=%d bytes, resized_size=%d bytes",
        len(image_bytes), len(resized),
    )

    # 3回試行して多数決
    votes: list[str] = []
    for i in range(3):
        vote = _single_classify(image_b64, vlm_model=vlm_model)
        votes.append(vote)
        logger.info("[Step0] Trial %d: %s", i + 1, vote)

    # 多数決（UNKNOWN以外が1票でもあればそちらを優先）
    from collections import Counter
    counts = Counter(votes)
    # UNKNOWN以外の票がある場合、その中で最多を採用
    non_unknown = {k: v for k, v in counts.items() if k != "UNKNOWN"}
    if non_unknown:
        result = max(non_unknown, key=non_unknown.get)
    else:
        result = "UNKNOWN"

    logger.info("[Step0] Votes=%s -> Final: %s", votes, result)
    return result


def step1_analyze_image(image_bytes: bytes, modality: str, vlm_model: str = VLM_MODEL) -> str:
    """ステップ1: VLMによる画像解析（動的プロンプト注入付き）."""
    resized = resize_image(image_bytes)
    checklist_info = MODALITY_CHECKLIST.get(modality, {})
    checklist_injection = (
        f"\n\n【重点チェックリスト】{checklist_info.get('checklist', '')}"
        if checklist_info else ""
    )
    
    return call_llm(
        prompt=(
            "この医療画像に異常所見がないか積極的に探してください。"
            "微細な濃度差・陰影・パターンの変化も見逃さず評価してください。"
            "チェックリストの各項目について必ず言及し、"
            "異常・正常のいずれであっても根拠を示して報告してください。"
            "【重要ルール】チェックリストの項目で1つでも陽性（異常疑い）所見があれば、"
            "総合印象は必ず「異常疑い」とすること。"
            "全18項目が完全に正常と確認できた場合のみ「正常」と判定してください。"
        ),
        system=(
            "You are an experienced radiologist with expertise in detecting subtle abnormalities. "
            "Your job is to actively look for pathological findings, not to confirm normality. "
            "Even subtle or borderline findings should be reported. "
            "CRITICAL RULE: If ANY single checklist item shows a positive/abnormal finding, "
            "the final impression MUST be 'abnormal' (異常疑い). "
            "Only conclude 'normal' (正常) when ALL items are definitively normal. "
            "Provide a systematic and detailed report in Japanese."
            f"{checklist_injection}"
        ),
        images=[base64.b64encode(resized).decode("utf-8")],
        model=vlm_model,
    )


def step2_search_guidelines(findings: str) -> str:
    """ステップ2: Agentic RAG - 所見に基づくガイドライン検索.

    LLMに所見を読み込ませ、知識ベースから適切なガイドラインを
    自律的に選択・抽出させる。
    """
    guidelines_text = "\n".join(
        f"- {key}: {g.get('condition', '')} | 対応: {g.get('action', '')} | 緊急度: {g.get('urgency', '')}"
        for key, g in CLINICAL_GUIDELINES.items()
    )
    
    prompt = (
        f"以下の画像所見の【個別項目の内容】を注意深く読み、最も関連するガイドラインを選択して説明してください。\n\n"
        f"【重要】「総合評価」「総合印象」などのラベル文字列は参考にしないでください。"
        f"個々の所見項目（陽性・陰性の内容）だけを根拠にガイドラインを選択すること。\n\n"
        f"所見:\n{findings}\n\nガイドライン:\n{guidelines_text}"
    )

    return call_llm(
        prompt=prompt,
        system=(
            "あなたは臨床判断支援エージェントです。"
            "所見の『総合評価』ラベルに引きずられず、各項目の実際の内容を根拠に判断してください。"
            "陽性所見が1つでもあれば、それに対応するガイドラインを優先的に選択してください。"
        )
    )


def step2b_search_similar_cases(findings: str) -> str:
    cases_text = "\n".join(
        f"- {c['case_id']}: {c['diagnosis']} | 所見: {c['findings']} | 治療: {c['treatment']}"
        for c in CASE_DATABASE
    )
    
    prompt = f"以下の所見に類似した過去症例を参照して、参考情報をまとめてください。\n\n所見:\n{findings}\n\n類似症例:\n{cases_text}"
    
    return call_llm(
        prompt=prompt,
        system="あなたは臨床判断支援エージェントです。類似症例を参考に実践的なアドバイスをしてください。"
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
            "あなたは臨床判断支援エージェントです。より広範な検索を実行しています。"
            "鑑別診断や関連疾患も含めて選択してください。"
            "必ず日本語で回答してください。"
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
            "あなたは臨床症例検索エージェントです。より広範な検索を実行しています。"
            "鑑別診断の参考になる症例も含めて選択してください。"
            "必ず日本語で回答してください。"
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

    # 日本語フォント設定（OS別に候補を探索）
    font_candidates = [
        "/Library/Fonts/Arial Unicode.ttf",                          # macOS 標準（最優先）
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",           # macOS フォールバック
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",   # Linux 別パス
    ]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)
    if font_path is None:
        raise RuntimeError("日本語フォントが見つかりません。Arial Unicode または Noto Sans CJK をインストールしてください。")
    pdf.add_font("JapaneseFont", "", font_path)
    pdf.set_font("JapaneseFont", size=10)

    # 有効描画幅を事前計算
    content_width = pdf.w - pdf.l_margin - pdf.r_margin

    # ヘッダー
    pdf.set_font("JapaneseFont", size=16)
    pdf.cell(content_width, 12, "診断支援レポート", align="C")
    pdf.ln(14)
    pdf.set_font("JapaneseFont", size=8)
    pdf.cell(
        content_width, 6,
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | VLM: {VLM_MODEL} | Text: {TEXT_MODEL}",
        align="C",
    )
    pdf.ln(12)

    # 本文（Markdown除去済み）
    clean_report = _strip_markdown(report)
    pdf.set_font("JapaneseFont", size=10)
    for line in clean_report.split("\n"):
        if line.strip() == "":
            pdf.ln(3)
        elif line.startswith("■"):
            pdf.ln(4)
            pdf.set_font("JapaneseFont", size=12)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(content_width, 8, line.replace("■", ""), align="L")
            pdf.set_font("JapaneseFont", size=10)
        else:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(content_width, 6, line, align="L")

    # 免責事項
    pdf.ln(10)
    pdf.set_font("JapaneseFont", size=8)
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

    # --- サイドバー: VLMモデル選択 ---
    with st.sidebar:
        st.header("⚙️ モデル設定")
        model_ids = [m["id"] for m in VLM_MODEL_OPTIONS]
        model_labels = [
            f"{m['label']}  ({m['size']} / RAM {m['ram']})"
            for m in VLM_MODEL_OPTIONS
        ]
        default_idx = model_ids.index(VLM_MODEL) if VLM_MODEL in model_ids else 0
        selected_idx = st.selectbox(
            "VLM（画像解析モデル）",
            range(len(model_ids)),
            format_func=lambda i: model_labels[i],
            index=default_idx,
            key="vlm_selector",
        )
        selected_vlm = model_ids[selected_idx]
        selected_info = VLM_MODEL_OPTIONS[selected_idx]
        st.caption(selected_info["desc"])

        # モデルが変わったらウォームアップをリセット
        if st.session_state.get("vlm_model") != selected_vlm:
            st.session_state["vlm_model"] = selected_vlm
            st.session_state.pop("vlm_warmed_up", None)

        st.divider()
        st.caption(f"テキストモデル: `{TEXT_MODEL}`")

        # ollama pull コマンドの案内
        with st.expander("モデルの取得方法"):
            st.code(f"ollama pull {selected_vlm}", language="bash")
    # サイドバーで選択されたモデルを使用
    active_vlm = st.session_state.get("vlm_model", VLM_MODEL)

    st.caption(
        f"VLM: `{active_vlm}` | Text: `{TEXT_MODEL}` | Runtime: Ollama | Agentic RAG Pipeline"
    )

    # Ollamaヘルスチェック
    if not check_ollama_health():
        st.error(
            "⚠️ Ollamaサーバーに接続できません。\n\n"
            "```bash\nollama serve\n```\n\n"
            "を実行してからページをリロードしてください。"
        )
        st.stop()

    st.success("✅ Ollama接続確認済み")

    # VLMモデルのウォームアップ（モデル変更時も再実行）
    if "vlm_warmed_up" not in st.session_state:
        with st.spinner(f"🔄 {active_vlm} をロード中（初回のみ、数分かかります）..."):
            if warmup_model(active_vlm):
                st.session_state["vlm_warmed_up"] = True
            else:
                st.error("VLMモデルのロードに失敗しました。Ollamaの状態を確認してください。")
                st.stop()

    # 画像アップロード
    uploaded_file = st.file_uploader(
        "医療画像をアップロード",
        type=SUPPORTED_FORMATS,
        help="X線・CT・MRI等の画像ファイル (PNG/JPEG)",
    )

    # アップロード画像をsession_stateに保存（セッション切断対策）
    if uploaded_file is not None:
        st.session_state["image_bytes"] = uploaded_file.getvalue()
        st.session_state["image_name"] = uploaded_file.name

    if "image_bytes" not in st.session_state:
        st.info("👆 画像ファイルを選択してください。")
        return

    # レイアウト: 左=画像、右=解析結果（タブ切替）
    col_image, col_result = st.columns([1, 2])

    with col_image:
        st.subheader("📷 アップロード画像")
        st.image(st.session_state["image_bytes"], use_container_width=True)

    with col_result:
        if st.button("🔍 エージェント解析を実行", type="primary", use_container_width=True):
            image_bytes = st.session_state["image_bytes"]

            try:
                # ステップ0: モダリティ判定（ルーター）
                with st.spinner("🏷️ Step 0/5: 画像モダリティを自律判定中..."):
                    modality = step0_classify_modality(image_bytes, vlm_model=active_vlm)
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
                    findings = step1_analyze_image(image_bytes, modality, vlm_model=active_vlm)
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
                    f"VLM: {VLM_MODEL} | Text: {TEXT_MODEL}"
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
