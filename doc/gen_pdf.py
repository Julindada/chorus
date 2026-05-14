"""Generate Chorus values questionnaire PDF (Chinese, 4 pages)."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak,
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# ── fonts ────────────────────────────────────────────────────────────────────
FONT_REGULAR = "STHeiti-Light"
FONT_BOLD    = "STHeiti-Medium"

pdfmetrics.registerFont(TTFont(FONT_REGULAR, "/System/Library/Fonts/STHeiti Light.ttc"))
pdfmetrics.registerFont(TTFont(FONT_BOLD,    "/System/Library/Fonts/STHeiti Medium.ttc"))

OUT_PATH = os.path.join(os.path.dirname(__file__), "chorus_values_questionnaire.pdf")

W, H = A4
MARGIN = 20 * mm

doc = SimpleDocTemplate(
    OUT_PATH,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN,  bottomMargin=MARGIN,
)


def S(name, font=FONT_REGULAR, size=10, leading=None, align=TA_LEFT, **kw):
    kw.setdefault("alignment", align)
    return ParagraphStyle(
        name, fontName=font, fontSize=size,
        leading=leading or size * 1.5,
        **kw
    )


s_title    = S("title",    FONT_BOLD,    20, align=TA_CENTER, spaceAfter=4)
s_subtitle = S("subtitle", FONT_REGULAR, 12, align=TA_CENTER, spaceAfter=10, textColor=colors.HexColor("#555555"))
s_h2       = S("h2",       FONT_BOLD,    13, spaceAfter=4,  spaceBefore=14)
s_h3       = S("h3",       FONT_BOLD,    11, spaceAfter=3,  spaceBefore=10)
s_body     = S("body",     FONT_REGULAR, 10, spaceAfter=4,  leading=16, align=TA_JUSTIFY)
s_dim      = S("dim",      FONT_REGULAR,  9, textColor=colors.HexColor("#666666"), spaceAfter=2)
s_q        = S("q",        FONT_REGULAR, 10, spaceAfter=2,  leading=15)
s_caption  = S("caption",  FONT_REGULAR,  8, align=TA_CENTER, textColor=colors.HexColor("#888888"))

SCALE_LABELS = "1　　2　　3　　4　　5　　6"

QUESTIONS = [
    "思考新想法、富有创意对他/她来说很重要。他/她喜欢用自己独特的方式做事。",
    "富有对他/她来说很重要。他/她希望拥有大量金钱和昂贵的东西。",
    "他/她认为世界上每个人都应该被平等对待。他/她相信每个人都应该有平等的机会。",
    "展示自己的能力对他/她很重要。他/她希望别人欣赏他/她所做的事情。",
    "生活在安全的环境中对他/她很重要。他/她会避免任何可能危及自身安全的事。",
    "他/她喜欢惊喜，总是寻找新事物去做。他/她认为在生活中尝试各种不同的事情很重要。",
    "他/她认为人们应该服从指令。他/她认为无论何时、哪怕无人监督，也应该遵守规则。",
    "聆听与自己不同的人对他/她很重要。即使不同意，他/她仍然想要理解对方。",
    "谦逊低调对他/她很重要。他/她尽量不引人注目。",
    "玩得开心对他/她很重要。他/她喜欢犒劳自己。",
    "自己做决定对他/她很重要。他/她喜欢自由，不依赖他人。",
    "帮助身边的人对他/她非常重要。他/她关心周围人的幸福。",
    "取得很大成功对他/她很重要。他/她希望别人认可自己的成就。",
    "政府保障安全、抵御所有威胁对他/她很重要。他/她希望国家足够强大，能够保护公民。",
    "他/她寻求冒险，喜欢承担风险。他/她想过一种充满激情的生活。",
    "举止得体对他/她始终很重要。他/她希望避免做任何被人认为不妥的事。",
    "获得他人尊重对他/她很重要。他/她希望别人按他/她说的去做。",
    "对朋友忠诚对他/她很重要。他/她愿意全心投入到亲近的人身上。",
    "他/她坚信人们应该爱护自然。保护环境对他/她很重要。",
    "传统对他/她很重要。他/她尽量遵循家族或宗教流传下来的习俗。",
    "他/她随时寻找娱乐机会。做让他/她快乐的事情对他/她来说很重要。",
]

story = []

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: Title + Instructions
# ════════════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 10 * mm))
story.append(Paragraph("Chorus 价值观问卷", s_title))
story.append(Paragraph("基于 Schwartz 基本人类价值观量表（ESS Round 11）", s_subtitle))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=8))

story.append(Paragraph(
    "这份问卷将帮助你建立个人价值观档案，用于 Chorus 多 Agent 决策系统的分析。"
    "共 21 道题，约需 5 分钟完成。",
    s_body
))

story.append(Spacer(1, 6 * mm))
story.append(Paragraph("使用说明", s_h2))
story.append(Paragraph(
    "下面将描述一些人物。请阅读每段描述，判断这个人有多像你，并在对应数字上画圈。",
    s_body
))

story.append(Spacer(1, 4 * mm))
story.append(Paragraph("评分标准", s_h2))

scale_data = [
    ["1", "2", "3", "4", "5", "6"],
    ["非常像我", "像我", "有点像我", "有一点像我", "不像我", "完全不像我"],
]
scale_table = Table(
    scale_data,
    colWidths=[(W - 2 * MARGIN) / 6] * 6,
    rowHeights=[10 * mm, 8 * mm],
)
scale_table.setStyle(TableStyle([
    ("FONTNAME",    (0, 0), (-1, -1), FONT_REGULAR),
    ("FONTNAME",    (0, 0), (-1, 0),  FONT_BOLD),
    ("FONTSIZE",    (0, 0), (-1, 0),  16),
    ("FONTSIZE",    (0, 1), (-1, 1),  9),
    ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#f0f0f0")),
    ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, 1), [colors.white]),
]))
story.append(scale_table)
story.append(Spacer(1, 3 * mm))
story.append(Paragraph(
    "1 = 非常像我　　2 = 像我　　3 = 有点像我　　4 = 有一点像我　　5 = 不像我　　6 = 完全不像我",
    s_dim
))

story.append(Spacer(1, 6 * mm))
story.append(Paragraph("填写须知", s_h2))
for note in [
    "每道题只选一个数字画圈，不要跳题。",
    "请根据第一直觉作答，无需过多思考。",
    "完成后按第四页的说明，将答案输入 Chorus 或交给 AI 换算。",
]:
    story.append(Paragraph(f"• {note}", s_body))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 + 3: Questions
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("价值观题目", s_h2))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"), spaceAfter=6))

COL_W   = W - 2 * MARGIN
SCORE_W = 42 * mm
TEXT_W  = COL_W - SCORE_W - 4 * mm

for i, q_text in enumerate(QUESTIONS, start=1):
    q_data = [[
        Paragraph(f"<b>{i}.</b>　{q_text}", s_q),
        Paragraph(SCALE_LABELS, ParagraphStyle(
            f"scale{i}", fontName=FONT_BOLD, fontSize=11,
            leading=16, alignment=TA_CENTER,
        )),
    ]]
    q_table = Table(q_data, colWidths=[TEXT_W, SCORE_W])
    q_table.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (0, 0),   0),
        ("RIGHTPADDING", (0, 0), (0, 0),   2),
        ("LEFTPADDING",  (1, 0), (1, 0),   2),
        ("RIGHTPADDING", (1, 0), (1, 0),   0),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LINEBELOW",    (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
    ]))
    story.append(q_table)

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: Two paths to get the value vector
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("获取价值观档案的两种方式", s_h2))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"), spaceAfter=8))

# ── Path A: Chorus program ────────────────────────────────────────────────
story.append(Paragraph("方式一：直接输入 Chorus（推荐）", s_h3))
story.append(Paragraph(
    "启动 Chorus 后，当系统提示建立价值观档案时，选择「问卷答案（推荐）」，"
    "按提示输入每道题的数字（1–6）。程序会自动换算并保存价值观档案。",
    s_body
))
story.append(Paragraph(
    "也可以把 21 个答案写在一行，用空格分隔，一次性粘贴：",
    s_body
))
story.append(Paragraph(
    "3  5  2  4  1  3  2  1  4  2  1  2  3  2  4  3  5  1  2  4  2",
    ParagraphStyle("oneline", fontName=FONT_BOLD, fontSize=10, leading=14,
                   backColor=colors.HexColor("#f0f4ff"), borderPadding=8,
                   spaceAfter=4, alignment=TA_CENTER)
))
story.append(Paragraph("（按题目顺序 1–21，每个数字之间用空格隔开）", s_dim))

story.append(Spacer(1, 6 * mm))
story.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#eeeeee"), spaceAfter=6))

# ── Path B: AI prompt ─────────────────────────────────────────────────────
story.append(Paragraph("方式二：交给 AI 计算（适合不运行程序时）", s_h3))
story.append(Paragraph(
    "完成问卷后，把下方提示词连同你的答案一起发给任意 AI（如 Claude、ChatGPT），"
    "AI 会直接输出可粘贴进 Chorus 的 JSON。",
    s_body
))

story.append(Spacer(1, 2 * mm))

ai_prompt_lines = [
    "我完成了一份 Schwartz 价值观问卷（ESS Round 11，21题，1–6分制）。",
    "请根据以下答案计算我的价值观向量，输出 JSON，键名和取值范围如下：",
    "",
    "键名：self_direction, stimulation, hedonism, achievement, power,",
    "      security, conformity, tradition, benevolence, universalism",
    "取值：0.0–1.0（公式：(6 − 答案) ÷ 5，多题取平均）",
    "",
    "题目与维度对应关系：",
    "1→self_direction, 2→power, 3→universalism, 4→achievement,",
    "5→security, 6→stimulation, 7→conformity, 8→universalism,",
    "9→tradition, 10→hedonism, 11→self_direction, 12→benevolence,",
    "13→achievement, 14→security, 15→stimulation, 16→conformity,",
    "17→power, 18→benevolence, 19→universalism, 20→tradition, 21→hedonism",
    "",
    "我的答案（题1–21）：",
    "题1:_  题2:_  题3:_  题4:_  题5:_  题6:_  题7:_",
    "题8:_  题9:_  题10:_ 题11:_ 题12:_ 题13:_ 题14:_",
    "题15:_ 题16:_ 题17:_ 题18:_ 题19:_ 题20:_ 题21:_",
    "",
    "只输出 JSON，不要解释。",
]
story.append(Paragraph(
    "<br/>".join(line.replace(" ", "&nbsp;") if line else "&nbsp;" for line in ai_prompt_lines),
    ParagraphStyle("aiprompt", fontName=FONT_REGULAR, fontSize=8.5, leading=13,
                   backColor=colors.HexColor("#f5f5f5"), borderPadding=8, spaceAfter=4)
))
story.append(Paragraph("填入答案后，将以上全部文字复制给 AI 即可。", s_dim))

story.append(Spacer(1, 6 * mm))
story.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#eeeeee"), spaceAfter=6))

# ── Path C: JSON paste ────────────────────────────────────────────────────
story.append(Paragraph("方式三：已有 JSON，直接粘贴进 Chorus", s_h3))
story.append(Paragraph(
    "若 AI 已输出 JSON，启动 Chorus 后选择「粘贴 JSON」，将 JSON 整体粘贴即可。",
    s_body
))

story.append(Spacer(1, 8 * mm))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"), spaceAfter=4))
story.append(Paragraph(
    "问卷基于 ESS Round 11 Human Values Scale（Schwartz, 1992）改编，仅供 Chorus 系统使用。",
    s_caption
))

# ── build ────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF saved: {OUT_PATH}")
