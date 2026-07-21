# pip install playwright
# playwright install chromium
#
# PITFALL (has bitten every helper here that edits a cell's style): a rendered
# table cell can wrap its label in a styled element, e.g.
# `<th ...><span style="font-style:italic">LN</span></th>`. Searching the whole
# cell string for `style="` matches that INNER span, so any style you inject
# (background, border, ...) lands on the span and tints/borders only part of the
# cell. When adding a style to a cell, split off the opening <th>/<td ...> tag
# with `re.match(r"<(th|td)\b[^>]*>", cell)` and modify ONLY that tag, never the
# nested markup. See paint_cell / add_left_border for the pattern.

import asyncio
import re
from pathlib import Path
from playwright.async_api import async_playwright


def stripe_rows_html(html, color="#f5f5f5", other="",
                     stripe_data_normal=True, index_cols=2):
    """Zebra-stripe a rendered HTML table, alternating *independently per column*.

    Works on the *final* HTML (after any header merging, tied-rowspan sub-rows,
    or group-header injection). Rather than shading whole physical ``<tr>`` rows
    -- which can't stay consistent once a column is rowspanned across rows of
    different shades -- we shade **each column on its own counter**: walking top
    to bottom, every time a real cell occupies a column its shade flips. A cell
    spanning several rows (an ``Architecture`` group, a tied value block) counts
    once, so that column alternates per group while a plain column beside it
    alternates per row. Columns are tracked on a grid so ``rowspan``/``colspan``
    shifts don't misalign the counters.

    stripe_data_normal, index_cols
        With ``stripe_data_normal`` (default), only the first ``index_cols``
        columns (Architecture, Dataset) use that per-column group alternation;
        every column beyond them is instead striped by *physical row* -- an
        ordinary zebra that flips on each ``<tr>`` (tied sub-rows included) and
        stays consistent across the whole data block. Set ``stripe_data_normal``
        False to group-alternate every column.

    The background is written directly onto each cell via an inline ``style``, so
    it survives to the PDF and doesn't depend on Styler's generated CSS classes.
    A cell spanning multiple columns takes the shade of its left-most column.
    Cells that already declare a ``background-color`` (explicit gradient/colour
    cells, e.g. the tied LN/RMS values) are left untouched, so those colours win.

    Pitfalls this function has to work around (all learned the hard way; see the
    inline comments for the exact handling):

    * pandas emits explicit cell colours in a ``<style>`` block keyed by cell id
      (``#T_x_row0_col0 { background-color: ... }``), NOT as inline styles. A
      naive "does the tag have a background?" check misses them, and because an
      inline stripe outranks those id selectors, striping would silently clobber
      the real colours. We pre-scan the ``<style>`` block for coloured ids.
    * a cell's rule may set ``background-color`` twice (e.g. ``#000`` then
      ``transparent`` for the ``--`` placeholders); the *last* wins, so only a
      non-transparent effective colour counts as "already coloured".
    * index/header cells can contain their own styled markup (the italic
      ``(#atoms)`` span); painting must touch only the cell's opening tag, never
      a nested ``style="..."``, or the tint lands on the inner span alone.
    """
    m = re.search(r"<tbody>(.*?)</tbody>", html, re.S)
    if not m:
        return html

    # Cells that pandas already coloured carry their background in the <style>
    # block, keyed by cell id (e.g. `#T_x_row0_col0 { background-color: ... }`),
    # NOT as an inline style on the tag. Collect those ids so we never overwrite
    # an explicit colour -- and note an inline stripe would otherwise *win*, since
    # inline styles outrank the id selectors pandas emits.
    coloured_ids = set()
    style_block = re.search(r"<style\b[^>]*>(.*?)</style>", html, re.S)
    if style_block:
        for rule in re.finditer(r"([^{}]+)\{([^{}]*)\}", style_block.group(1)):
            # A rule may declare background-color more than once (pandas emits e.g.
            # `background-color: #000; ...; background-color: transparent` for the
            # '--' placeholder cells); the *last* declaration wins in CSS. Only a
            # real, non-transparent effective colour counts as "already coloured"
            # -- transparent cells stay eligible for striping.
            decls = re.findall(r"background-color:\s*([^;]+)", rule.group(2))
            if decls and decls[-1].strip().lower() not in ("transparent", "none"):
                coloured_ids.update(re.findall(r"#([\w-]+)", rule.group(1)))

    def paint_cell(cell, bg):
        # Operate ONLY on the cell's opening <th>/<td ...> tag -- never on nested
        # markup. Cell content can carry its own styled elements (e.g. the italic
        # "(#atoms)" span, <span style="font-style:italic">), and a naive
        # style="..." search would inject the background into that inner span
        # instead of the cell, tinting only part of the cell.
        mo = re.match(r"<(th|td)\b[^>]*>", cell)
        open_tag, rest = mo.group(0), cell[mo.end():]
        cid = re.search(r'id="([^"]+)"', open_tag)
        already_coloured = (
            "background-color" in open_tag
            or (cid and cid.group(1) in coloured_ids)
        )
        if not bg or already_coloured:
            return cell  # keep an explicit cell colour; white band adds nothing
        if 'style="' in open_tag:
            open_tag = open_tag.replace('style="', f'style="background-color: {bg}; ', 1)
        else:
            open_tag = re.sub(r"^<(th|td)\b",
                              rf'<\1 style="background-color: {bg}"', open_tag)
        return open_tag + rest

    def span(cell, attr):
        mm = re.search(rf'{attr}="(\d+)"', cell)
        return int(mm.group(1)) if mm else 1

    rows = re.findall(r"<tr>.*?</tr>", m.group(1), re.S)
    col_parity = {}          # column index -> how many cells have started there
    pending = {}             # column index -> remaining rowspan rows still occupied
    out_rows = []
    for row_i, row in enumerate(rows):
        cells = re.findall(r"<(?:th|td)\b.*?</(?:th|td)>", row, re.S)
        painted, ci = [], 0
        for cell in cells:
            while pending.get(ci, 0) > 0:   # skip columns held by a rowspan above
                ci += 1
            cs, rs = span(cell, "colspan"), span(cell, "rowspan")
            if stripe_data_normal and ci >= index_cols:
                # data columns: plain per-physical-row zebra
                parity = row_i % 2
            else:
                # index columns: flip on each cell that *starts* here (per group)
                col_parity[ci] = col_parity.get(ci, 0) + 1
                parity = 1 - (col_parity[ci] % 2)
            bg = color if parity == 0 else other
            painted.append(paint_cell(cell, bg))
            for c in range(ci, ci + cs):    # reserve spanned rows for later rows
                if rs > 1:
                    pending[c] = rs
            ci += cs
        for c in list(pending):             # consume one row of every active rowspan
            pending[c] -= 1
            if pending[c] <= 0:
                del pending[c]
        # reassemble the row, preserving its original <tr ...> tag
        head = re.match(r"<tr[^>]*>", row).group(0)
        out_rows.append(head + "".join(painted) + "</tr>")
    new_tbody = "<tbody>" + "".join(out_rows) + "</tbody>"
    return html[:m.start()] + new_tbody + html[m.end():]


def border_second_header_row(html, color="#ececec", width="1px"):
    """Draw thin vertical separators between the column-name (2nd) header cells,
    except *inside* a spanning group-header from the row above it.

    A two-tier header (group row on top, per-column names below) reads better
    with light rules between the lower cells -- but a rule inside a group span
    (e.g. between ``LN`` and ``RMS`` under one ``I-SchNet`` header, or between the
    ``1..8`` layer columns under ``Explicit``) fights the grouping. So we border
    every lower cell boundary *except* those that fall within a real group span.

    Group membership is read from the top header row's ``colspan`` layout: a
    ``level_group`` cell spanning >1 column marks its trailing columns as
    "interior" (no left border); a ``blank`` spacer does not group, so its columns
    still get separated. No-op unless the ``<thead>`` has two rows (i.e. a group
    header was injected). Border colour defaults to the top header row's shade.
    """
    thead_m = re.search(r"<thead>(.*?)</thead>", html, re.S)
    if not thead_m:
        return html
    trs = re.findall(r"<tr.*?</tr>", thead_m.group(1), re.S)
    if len(trs) < 2:
        return html  # single (merged) header row -> nothing to separate under

    def cells(tr):
        return re.findall(r"<(?:th|td)\b.*?</(?:th|td)>", tr, re.S)

    def span(cell, attr):
        mm = re.search(rf'{attr}="(\d+)"', cell)
        return int(mm.group(1)) if mm else 1

    # Walk the top row and mark, per column, whether it is *interior* to a real
    # (non-blank) group span -> those lower cells get no left border.
    interior = {}
    col = 0
    for cell in cells(trs[0]):
        cs = span(cell, "colspan")
        is_group = "level_group" in cell and cs > 1
        for k in range(cs):
            interior[col + k] = is_group and k > 0  # first col of a group still bordered
        col += cs

    # Add the left border to lower-row cells that are not interior (skip col 0,
    # the table's outer edge).
    def add_left_border(cell):
        # Touch only the opening tag -- a header cell can wrap its label in a
        # styled <span> (the italic 'LN'/'RMS'/'(#atoms)'), and matching 'style="'
        # anywhere would put the border on that inner span instead of the cell.
        style = f"border-left: {width} solid {color}"
        mo = re.match(r"<(th|td)\b[^>]*>", cell)
        open_tag, rest = mo.group(0), cell[mo.end():]
        if 'style="' in open_tag:
            open_tag = open_tag.replace('style="', f'style="{style}; ', 1)
        else:
            open_tag = re.sub(r"^<(th|td)\b", rf'<\1 style="{style}"', open_tag)
        return open_tag + rest

    lower = trs[1]
    out, col = [], 0
    for cell in cells(lower):
        cs = span(cell, "colspan")
        if col > 0 and not interior.get(col, False):
            cell = add_left_border(cell)
        out.append(cell)
        col += cs
    head = re.match(r"<tr[^>]*>", lower).group(0)
    new_lower = head + "".join(out) + "</tr>"
    new_thead = thead_m.group(0).replace(trs[1], new_lower, 1)
    return html[:thead_m.start()] + new_thead + html[thead_m.end():]


# Minimal CSS to reproduce how Jupyter renders a pandas Styler. The Styler's own
# inline styles (background_gradient colors, text alignment from set_properties,
# set_table_styles) are emitted by to_html() and take precedence over these
# defaults, so we only supply the structural look Jupyter adds on top of the
# bare HTML table. Keep this list-style/border setup close to nbconvert's
# default dataframe styling.
_NOTEBOOK_TABLE_CSS = """
    #content { display: inline-block; }  /* crop the page box to the table */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 12px;
    }
    table { border-collapse: collapse; border: none; }
    /* Jupyter shades the header band darker than the body zebra, and the bottom
       header row (column names) a shade darker than the top group-header row --
       its own to_html() doesn't emit either, so reproduce them here. The two
       header rows are Styler's group-header (top) + column/index-name row
       (bottom); target the bottom via last-child so it overrides the base. */
    table thead th { text-align: right; vertical-align: bottom; background-color: #ececec; }
    table thead tr:last-child th { background-color: #dcdcdc; }
    table th, table td { padding: 0.2em 0.5em; line-height: normal; }
    table tbody th { font-weight: 400; text-align: right; vertical-align: middle; }
"""


async def styler_html_to_pdf_async(
    styled,
    pdf_path="table.pdf",
    *,
    landscape=False,
    scale=1.0,
    margin_mm=0,
    extra_css="",
    stripe=True,
    header_borders=True,
    **stripe_kwargs
):
    """Render a pandas Styler to a tightly-cropped PDF, matching the notebook look.

    The Styler is serialized with ``to_html()`` (which carries the gradient
    colors as inline styles) and rendered with headless Chromium. We do not
    rewrite the header or force any alignment, so the output mirrors what Jupyter
    displays. The PDF page is sized to exactly the table's bounding box.

    Parameters
    ----------
    scale, margin_mm
        ``margin_mm`` adds whitespace around the cropped table; ``scale`` scales
        the rendered content (Chromium's pdf scale, 0.1-2).
    landscape
        Accepted for backward compatibility but ignored: the page is sized to the
        table's exact bounding box, so orientation has no meaning (and passing it
        alongside an explicit width/height corrupts Chromium's layout).
    extra_css
        Optional CSS appended after the defaults (e.g. to tweak fonts/borders).
    stripe
        Zebra-stripe the rendered rows (Jupyter's alternating shading, which its
        own to_html() doesn't emit). Applied on the final HTML so it aligns with
        the actual rows even for merged-header / tied-rowspan tables. Pass False
        to disable; for custom shades call ``stripe_rows_html`` yourself.
    header_borders
        Draw light vertical rules between the column-name header cells, skipping
        those inside a spanning group header (see ``border_second_header_row``).
        No-op unless a group-header row is present.
    """
    html = styled.to_html()
    if stripe:
        html = stripe_rows_html(html, **stripe_kwargs)
    if header_borders:
        html = border_second_header_row(html)

    wrapped = f"""<!doctype html>
        <html>
        <head>
        <meta charset="utf-8" />
        <style>
        @page {{ margin: {margin_mm}mm; }}
        html, body {{ margin: 0; padding: 0; }}
        {_NOTEBOOK_TABLE_CSS}
        {extra_css}
        </style>
        </head>
        <body>
        <div id="content">{html}</div>
        </body>
        </html>"""

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    margin_px = f"{margin_mm}mm"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(wrapped, wait_until="networkidle")

        # measure the table under print media so the page is sized to fit it
        await page.emulate_media(media="print")
        w, h = await page.evaluate("""
        () => {
            const el = document.getElementById("content");
            const r = el.getBoundingClientRect();
            return [Math.ceil(r.width), Math.ceil(r.height)];
        }
        """)
        PAD = 2  # px safety padding; bump to 4 if a sliver spills to page 2
        s = scale if 0.1 <= scale <= 2 else 1.0  # Chromium's allowed pdf range

        # page.pdf scales content by `s`, but w/h were measured at scale 1, so
        # scale the page box too to keep the crop tight at any `scale`.
        await page.pdf(
            path=str(pdf_path),
            print_background=True,
            width=f"{w * s + PAD}px",
            height=f"{h * s + PAD}px",
            margin={"top": margin_px, "right": margin_px,
                    "bottom": margin_px, "left": margin_px},
            scale=s,
            page_ranges="1",
        )
        await browser.close()

    return pdf_path


def styler_html_to_pdf(styled, *args, **kwargs):
    # works in notebooks too
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(styler_html_to_pdf_async(styled, *args, **kwargs))
