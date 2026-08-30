from __future__ import annotations

import re
import unicodedata

SIMPLE_REPLACE_MAP: dict[str, str] = {
    "\t": "",
    "[n]": "",
    r"\[n\]": "",
    "　": "",
    "？": "?",
    "！": "!",
    "♥": "♡",
    "●": "○",
    "◯": "○",
    "〇": "○",
}

REGEX_REPLACE_MAP = {
    re.compile(r"[;▼♀♂《》≪≫①②③④⑤⑥]"): "",
    re.compile(r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]"): "",
    re.compile(r"[\uff5e\u301C]"): "ー",
    re.compile(r"…{3,}"): "……",
}

# NKFCで変換されない残余に備えた追加変換テーブル
FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
            strict=True,
        )
    }
)

_HALFWIDTH_KATAKANA_CHARS = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
_FULLWIDTH_KATAKANA_CHARS = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)

FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A), strict=True)
    }
)


def strip_outer_brackets(text: str) -> str:
    pairs = {"「": "」", "『": "』", "（": "）", "【": "】", "(": ")"}

    while True:
        if len(text) < 2:
            break

        start_char = text[0]
        end_char = text[-1]

        if start_char in pairs and pairs[start_char] == end_char:
            depth = 0
            is_enclosing_all = True

            for i, char in enumerate(text):
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1

                if depth == 0 and i < len(text) - 1:
                    is_enclosing_all = False
                    break

            if is_enclosing_all and depth == 0:
                text = text[1:-1]
                continue

        break

    return text


def normalize_text(text: str) -> str:
    for old, new in SIMPLE_REPLACE_MAP.items():
        text = text.replace(old, new)

    for pattern, replacement in REGEX_REPLACE_MAP.items():
        text = pattern.sub(replacement, text)

    text = strip_outer_brackets(text)

    text = unicodedata.normalize("NFKC", text)

    # NFKC後の残余に対する追加変換（半角カタカナはNFKCで全角化されるが念のため）
    text = text.translate(FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(HALFWIDTH_KATAKANA_TO_FULLWIDTH)

    text = text.replace("...", "…")
    text = text.replace("..", "…")

    text = re.sub(r"…{3,}", "……", text)

    if text.endswith(("。", "、")):
        text = text.rstrip("。、")

    return text
