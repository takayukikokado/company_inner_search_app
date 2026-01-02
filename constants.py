"""
このファイルは、固定の文字列や数値などのデータを変数として一括管理するファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

import re
import os
import csv
from typing import List, Dict, Any, Optional

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from langchain.docstore.document import Document  # type: ignore
        except Exception:  # pragma: no cover
            Document = None  # type: ignore


def _normalize_key(s: str) -> str:
    return re.sub(r"[\s_\-　]+", "", str(s)).lower()


def _pick_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    if not fieldnames:
        return None
    norm_map = {fn: _normalize_key(fn) for fn in fieldnames}
    cand_norm = [_normalize_key(c) for c in candidates]
    # exact/contains match in priority order
    for c in cand_norm:
        for fn, n in norm_map.items():
            if n == c:
                return fn
        for fn, n in norm_map.items():
            if c in n or n in c:
                return fn
    return None


def _read_csv_rows(file_path: str, encodings: List[str]) -> tuple[List[str], List[Dict[str, str]], str]:
    last_err = None
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = [fn.strip() for fn in (reader.fieldnames or []) if fn is not None]
                rows: List[Dict[str, str]] = []
                for r in reader:
                    row: Dict[str, str] = {}
                    for k, v in (r or {}).items():
                        if k is None:
                            continue
                        kk = str(k).strip()
                        vv = "" if v is None else str(v).strip()
                        row[kk] = vv
                    rows.append(row)
            return fieldnames, rows, enc
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("CSV read failed")


class EmployeeRosterCSVLoader:
    """
    社員名簿.csv のみを対象に、行分割ではなく 1 つの Document に統合して取り込むローダー。
    それ以外の CSV は従来どおり CSVLoader にフォールバックする。
    """

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self):
        base = os.path.basename(self.file_path)
        # 特定ファイル以外は従来ローダーを使用（影響範囲を最小化）
        if base.lower() != "社員名簿.csv":
            return CSVLoader(self.file_path, encoding=self.encoding).load()

        # 文字コードの揺れに耐える
        encodings = [self.encoding, "utf-8-sig", "cp932", "shift_jis", "utf-16"]
        try:
            fieldnames, rows, used_enc = _read_csv_rows(self.file_path, encodings)
        except Exception:
            # 最後の手段：従来ローダー
            return CSVLoader(self.file_path, encoding=self.encoding).load()

        if Document is None:
            # Document が import できない環境ではフォールバック
            return CSVLoader(self.file_path, encoding=used_enc).load()

        # 列推定（よくある表記揺れに対応）
        dept_key = _pick_column(fieldnames, ["所属部署", "部署", "部門", "所属", "department", "dept", "section", "team"])
        name_key = _pick_column(fieldnames, ["氏名", "名前", "社員名", "name", "fullname"])
        id_key = _pick_column(fieldnames, ["社員ID", "社員番号", "社員No", "id", "employeeid", "empid"])
        title_key = _pick_column(fieldnames, ["役職", "職種", "職位", "title", "job", "position"])

        used_keys = [k for k in [name_key, dept_key, title_key, id_key] if k]

        def fmt_row(r: Dict[str, str]) -> str:
            parts: List[str] = []
            if name_key and r.get(name_key):
                parts.append(f"氏名: {r.get(name_key)}")
            if dept_key and r.get(dept_key):
                parts.append(f"部署: {r.get(dept_key)}")
            if title_key and r.get(title_key):
                parts.append(f"役職: {r.get(title_key)}")
            if id_key and r.get(id_key):
                parts.append(f"社員ID: {r.get(id_key)}")

            # 追加情報（長くなりすぎないように最大 3 項目）
            extras = [k for k in fieldnames if k not in used_keys and r.get(k)]
            for k in extras[:3]:
                parts.append(f"{k}: {r.get(k)}")

            # 何も取れない場合は行全体を簡易表現
            if not parts:
                compact = " / ".join(f"{k}: {r.get(k,'')}" for k in fieldnames[:5] if k)
                return compact or ""
            return " / ".join(parts)

        # 部署ごとにまとめる（人事部を先頭に置くことで該当検索の再現性を上げる）
        groups: Dict[str, List[Dict[str, str]]] = {}
        if dept_key:
            for r in rows:
                dept = (r.get(dept_key, "") or "").strip() or "未設定"
                groups.setdefault(dept, []).append(r)
        else:
            groups["全社員"] = rows

        def dept_sort_key(d: str):
            # 人事系を最優先
            if "人事" in d:
                return (0, d)
            return (1, d)

        ordered_depts = sorted(groups.keys(), key=dept_sort_key)

        # 検索に強い形にテキストを整形（部署見出し + 1行1名の正規化）
        header_cols = "、".join(fieldnames) if fieldnames else "（不明）"
        content_lines: List[str] = []
        content_lines.append("社員名簿（CSV）")
        content_lines.append(f"ファイル: {base}")
        content_lines.append(f"列: {header_cols}")
        content_lines.append("")
        content_lines.append("【部署別の社員一覧】")
        content_lines.append("")

        for dept in ordered_depts:
            content_lines.append(f"### 部署: {dept}")
            # 1行1名で出力（同一部署の情報を近接させ、k=5でも複数名が拾われやすくする）
            for r in groups[dept]:
                line = fmt_row(r).strip()
                if line:
                    # 部署語を各行にも含め、埋め込みの手掛かりを強化
                    if dept_key and "部署:" not in line:
                        line = f"部署: {dept} / " + line
                    content_lines.append(f"- {line}")
            content_lines.append("")

        merged_text = "\n".join(content_lines).strip()

        doc = Document(
            page_content=merged_text,
            metadata={
                "source": self.file_path,
                "file_name": base,
                "file_type": "csv",
                "encoding": used_enc,
                "row_count": len(rows),
            },
        )
        return [doc]


############################################################
# 共通変数の定義
############################################################

# ==========================================
# 画面表示系
# ==========================================
APP_NAME = "社内情報特化型生成AI検索アプリ"
ANSWER_MODE_1 = "社内文書検索"
ANSWER_MODE_2 = "社内問い合わせ"
CHAT_INPUT_HELPER_TEXT = "こちらからメッセージを送信してください。"
DOC_SOURCE_ICON = ":material/description: "
LINK_SOURCE_ICON = ":material/link: "
WARNING_ICON = ":material/warning:"
ERROR_ICON = ":material/error:"
SPINNER_TEXT = "回答生成中..."


# ==========================================
# ログ出力系
# ==========================================
LOG_DIR_PATH = "./logs"
LOGGER_NAME = "ApplicationLog"
LOG_FILE = "application.log"
APP_BOOT_MESSAGE = "アプリが起動されました。"


# ==========================================
# LLM設定系
# ==========================================
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.5


# ==========================================
# RAG参照用のデータソース系
# ==========================================
RAG_TOP_FOLDER_PATH = "./data"
SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": lambda path: EmployeeRosterCSVLoader(path, encoding="utf-8"),
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
}
WEB_URL_LOAD_TARGETS = [
    "https://generative-ai.web-camp.io/"
]


# ==========================================
# プロンプトテンプレート
# ==========================================
SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT = "会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。"

SYSTEM_PROMPT_DOC_SEARCH = """
    あなたは社内の文書検索アシスタントです。
    以下の条件に基づき、ユーザー入力に対して回答してください。

    【条件】
    1. ユーザー入力内容と以下の文脈との間に関連性がある場合、空文字「""」を返してください。
    2. ユーザー入力内容と以下の文脈との関連性が明らかに低い場合、「該当資料なし」と回答してください。

    【文脈】
    {context}
"""

SYSTEM_PROMPT_INQUIRY = """
    あなたは社内情報特化型のアシスタントです。
    以下の条件に基づき、ユーザー入力に対して回答してください。

    【条件】
    1. ユーザー入力内容と以下の文脈との間に関連性がある場合のみ、以下の文脈に基づいて回答してください。
    2. ユーザー入力内容と以下の文脈との関連性が明らかに低い場合、「回答に必要な情報が見つかりませんでした。」と回答してください。
    3. 憶測で回答せず、あくまで以下の文脈を元に回答してください。
    4. できる限り詳細に、マークダウン記法を使って回答してください。
    5. マークダウン記法で回答する際にhタグの見出しを使う場合、最も大きい見出しをh3としてください。
    6. 複雑な質問の場合、各項目についてそれぞれ詳細に回答してください。
    7. 必要と判断した場合は、以下の文脈に基づかずとも、一般的な情報を回答してください。

    {context}
"""


# ==========================================
# LLMレスポンスの一致判定用
# ==========================================
INQUIRY_NO_MATCH_ANSWER = "回答に必要な情報が見つかりませんでした。"
NO_DOC_MATCH_ANSWER = "該当資料なし"


# ==========================================
# エラー・警告メッセージ
# ==========================================
COMMON_ERROR_MESSAGE = "このエラーが繰り返し発生する場合は、管理者にお問い合わせください。"
INITIALIZE_ERROR_MESSAGE = "初期化処理に失敗しました。"
NO_DOC_MATCH_MESSAGE = """
    入力内容と関連する社内文書が見つかりませんでした。\n
    入力内容を変更してください。
"""
CONVERSATION_LOG_ERROR_MESSAGE = "過去の会話履歴の表示に失敗しました。"
GET_LLM_RESPONSE_ERROR_MESSAGE = "回答生成に失敗しました。"
DISP_ANSWER_ERROR_MESSAGE = "回答表示に失敗しました。"
############################################################
# RAG設定
############################################################
# ベクターストアから取得してプロンプトに埋め込む関連ドキュメント数
RAG_RETRIEVER_K = 5
# チャンク分割サイズ
RAG_CHUNK_SIZE = 500
# チャンクの重なり（オーバーラップ）
RAG_CHUNK_OVERLAP = 50

# Q6