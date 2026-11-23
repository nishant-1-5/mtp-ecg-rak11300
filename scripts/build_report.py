"""Build docs/report/mtp_report.docx from docs/report/mtp_report.md.

    python scripts/build_report.py

Supports the Markdown subset the report uses: headings, paragraphs (with
**bold**, *italic*, `code`, <url>), bullet and numbered lists, pipe tables,
images with captions, and fenced code blocks. The .md stays the source of truth;
the .docx is for submission and further editing in Word.
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REPORT = Path(__file__).resolve().parents[1] / "docs" / "report"
INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`|<https?://[^>]+>)")


def add_runs(par, text):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif part.startswith("<http"):
            r = par.add_run(part[1:-1])
            r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x99)
            r.underline = True
        elif part.startswith("*"):
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    align = cells[1]
    body = [cells[0]] + cells[2:]
    t = doc.add_table(rows=len(body), cols=len(body[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(body):
        for j, txt in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            add_runs(p, txt)
            for r in p.runs:
                r.font.size = Pt(9)
                if i == 0:
                    r.bold = True
            if align[j].endswith(":"):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            if i == 0:
                shade(cell, "D9E2F3")
    doc.add_paragraph()


def build(md_path, out_path):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2)
        s.top_margin = s.bottom_margin = Cm(2.0)
    fig_no = 0
    i = 0
    para = []

    def flush():
        if para:
            p = doc.add_paragraph()
            add_runs(p, " ".join(x.strip() for x in para))
            p.paragraph_format.space_after = Pt(6)
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            r = p.add_run("\n".join(code))
            r.font.name = "Consolas"
            r.font.size = Pt(8.5)
            i += 1
            continue
        m = re.match(r"^(#{1,3}) (.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            if level == 1:
                h = doc.add_heading(m.group(2), 0)
            else:
                h = doc.add_heading(m.group(2), level - 1)
            i += 1
            continue
        m = re.match(r"^!\[(.*)\]\((.*)\)", line)
        if m:
            flush()
            fig_no += 1
            doc.add_picture(str(md_path.parent / m.group(2)), width=Cm(16.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cap.add_run(f"Figure {fig_no}. {m.group(1)}")
            r.italic = True
            r.font.size = Pt(9.5)
            i += 1
            continue
        if line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            add_table(doc, rows)
            continue
        m = re.match(r"^(\s*)(- |\d+\. )(.*)", line)
        if m:
            flush()
            numbered = m.group(2)[0].isdigit()
            text = [m.group(3)]
            i += 1
            while i < len(lines) and lines[i].startswith(" ") and lines[i].strip() \
                    and not re.match(r"^\s*(- |\d+\. )", lines[i]):
                text.append(lines[i].strip())
                i += 1
            if numbered:
                # Literal numbers: Word's "List Number" would continue numbering across lists.
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.9)
                p.paragraph_format.first_line_indent = Cm(-0.6)
                p.paragraph_format.space_after = Pt(3)
                add_runs(p, m.group(2) + " ".join(text))
            else:
                p = doc.add_paragraph(style="List Bullet")
                add_runs(p, " ".join(text))
            continue
        if not line.strip():
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    doc.save(out_path)
    print(f"wrote {out_path} ({fig_no} figures)")


if __name__ == "__main__":
    build(REPORT / "mtp_report.md", REPORT / "mtp_report.docx")
