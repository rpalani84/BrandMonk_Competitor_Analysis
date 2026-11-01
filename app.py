from datetime import datetime
from pathlib import Path
from io import BytesIO
from urllib.parse import urlparse

import json
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from xml.sax.saxutils import escape

BASE_DIR = Path(__file__).resolve().parent
FONT_DIR = BASE_DIR / "fonts"
pdfmetrics.registerFont(TTFont("NotoSans", str(FONT_DIR / "NotoSans-Regular.ttf")))
pdfmetrics.registerFont(TTFont("NotoSans-Bold", str(FONT_DIR / "NotoSans-Bold.ttf")))
registerFontFamily(
    "NotoSans", normal="NotoSans", bold="NotoSans-Bold",
    italic="NotoSans", boldItalic="NotoSans-Bold"
)

st.set_page_config(
    page_title="Competitor Analysis Agent",
    page_icon="\U0001F4CA",
    layout="wide",
)

WEIGHTS = {
    "Search visibility": 25,
    "Reputation": 20,
    "Paid activity": 15,
    "Keyword strategy": 15,
    "Social presence": 10,
    "Website quality": 10,
    "Local presence": 5,
}
METRIC_COLUMNS = list(WEIGHTS)


SCORE_STORE_PATH = Path(__file__).with_name("competitor_scores.json")


def load_competitor_score_cache():
    """Load saved scores from the separate local JSON file."""
    if not SCORE_STORE_PATH.exists():
        return {}

    try:
        records = json.loads(
            SCORE_STORE_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}

    cache = {}
    if not isinstance(records, list):
        return cache

    for record in records:
        if not isinstance(record, dict):
            continue

        competitor = str(record.get("competitor", "")).strip()
        website = str(record.get("website", "")).strip()
        scores = record.get("scores", {})

        if not competitor or not isinstance(scores, dict):
            continue

        valid_scores = {}
        for metric in METRIC_COLUMNS:
            value = scores.get(metric)
            if value is None or value == "":
                continue
            try:
                valid_scores[metric] = max(
                    0.0, min(100.0, float(value))
                )
            except (ValueError, TypeError):
                continue

        if valid_scores:
            cache[(competitor, website)] = valid_scores

    return cache


def save_competitor_score_cache(cache):
    """Atomically save nonblank scores without modifying the workbook."""
    records = []

    for (competitor, website), scores in sorted(cache.items()):
        valid_scores = {}

        for metric in METRIC_COLUMNS:
            value = scores.get(metric)
            if value is None or value == "":
                continue
            try:
                valid_scores[metric] = max(
                    0.0, min(100.0, float(value))
                )
            except (ValueError, TypeError):
                continue

        if competitor and valid_scores:
            records.append({
                "competitor": competitor,
                "website": website,
                "scores": valid_scores,
            })

    temporary_path = SCORE_STORE_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(SCORE_STORE_PATH)


def apply_competitor_score_cache(data, cache):
    """Overlay saved scores on the rows for the selected city."""
    result = data.copy()

    for index, row in result.iterrows():
        pair = (
            str(row.get("Competitor", "")).strip(),
            str(row.get("Website", "")).strip(),
        )
        saved = cache.get(pair, {})

        for metric, value in saved.items():
            if metric in METRIC_COLUMNS:
                result.at[index, metric] = value

    return result


def normalize_url(value):
    value = str(value or "").strip()
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        value = "https://" + value
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or "." not in parsed.netloc:
        return ""
    return value


def audit_website(url):
    result = {
        "Website": url,
        "Audit status": "Not checked",
        "HTTP status": None,
        "HTTPS": False,
        "Page title": "",
        "Meta description": "",
        "H1 count": 0,
        "H2 count": 0,
        "Images": 0,
        "Images missing alt": 0,
        "Word count": 0,
        "Website quality": None,
        "Audit notes": "",
        "Content warning": "",
    }

    if not url:
        result["Audit notes"] = "No valid website URL supplied."
        return result

    try:
        try:
            response = requests.get(
                url,
                timeout=12,
                headers={"User-Agent": "CompetitorAnalysisAgent/1.0"},
                allow_redirects=True,
            )
        except requests.Timeout:
            # Retry once for a slow public website.
            response = requests.get(
                url,
                timeout=20,
                headers={"User-Agent": "CompetitorAnalysisAgent/1.0"},
                allow_redirects=True,
            )
        result["HTTP status"] = response.status_code
        result["HTTPS"] = response.url.lower().startswith("https://")
        result["Website"] = response.url

        if response.status_code >= 400:
            result["Audit status"] = "HTTP error"
            result["Audit notes"] = f"Server returned HTTP {response.status_code}."
            return result

        soup = BeautifulSoup(response.text, "html.parser")
        result["Page title"] = soup.title.get_text(" ", strip=True) if soup.title else ""
        meta = soup.find("meta", attrs={"name": lambda value: value and value.lower() == "description"})
        result["Meta description"] = meta.get("content", "").strip() if meta else ""
        result["H1 count"] = len(soup.find_all("h1"))
        result["H2 count"] = len(soup.find_all("h2"))

        images = soup.find_all("img")
        result["Images"] = len(images)
        result["Images missing alt"] = sum(
            1 for image in images if not str(image.get("alt", "")).strip()
        )

        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        visible_text = soup.get_text(" ", strip=True)
        result["Word count"] = len(visible_text.split())

        # Detect common expired-domain messages without changing the technical score.
        page_content = " ".join([
            result["Page title"],
            result["Meta description"],
            visible_text[:12000],
        ]).lower()

        expired_domain_phrases = (
            "domain is expired",
            "domain has expired",
            "this domain has expired",
            "domain expired",
            "expired domain",
        )

        if any(phrase in page_content for phrase in expired_domain_phrases):
            result["Content warning"] = (
                "Possible expired-domain page - needs manual review."
            )

        score = 0
        score += 20 if result["HTTPS"] else 0
        score += 20 if result["Page title"] else 0
        score += 20 if result["Meta description"] else 0
        score += 15 if result["H1 count"] == 1 else (7 if result["H1 count"] > 1 else 0)
        score += 10 if result["H2 count"] > 0 else 0
        score += 10 if result["Word count"] >= 200 else (5 if result["Word count"] >= 50 else 0)
        score += 5 if result["Images"] == 0 or result["Images missing alt"] == 0 else 0

        result["Website quality"] = min(score, 100)
        result["Audit status"] = "OK"
        result["Audit notes"] = "Heuristic public-page audit; not a search ranking."
    except requests.RequestException as exc:
        result["Audit status"] = "Failed"
        result["Audit notes"] = f"Website request failed: {type(exc).__name__}"
    except Exception as exc:
        result["Audit status"] = "Failed"
        result["Audit notes"] = f"Audit failed: {type(exc).__name__}"
    return result


def calculate_score(row):
    weighted_total = 0.0
    weight_total = 0
    for metric, weight in WEIGHTS.items():
        value = row.get(metric)
        if pd.isna(value):
            continue
        try:
            value = max(0.0, min(100.0, float(value)))
        except (ValueError, TypeError):
            continue
        weighted_total += value * weight
        weight_total += weight
    score = weighted_total / weight_total if weight_total else None
    return score, weight_total

def build_pdf(business, industry, location, results, timestamp):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()

    for style_name in ("Title", "Heading2", "Heading3", "Normal", "BodyText"):
        styles[style_name].fontName = "NotoSans"

    styles["Title"].fontSize = 20
    styles["Title"].leading = 24

    meta_style = ParagraphStyle(
        "ReportMeta",
        parent=styles["Normal"],
        fontName="NotoSans",
        fontSize=10.5,
        leading=13,
    )

    intro_style = ParagraphStyle(
        "ReportIntro",
        parent=styles["BodyText"],
        fontName="NotoSans",
        fontSize=10,
        leading=13,
    )

    audit_intro_style = ParagraphStyle(
        "AuditIntro",
        parent=styles["BodyText"],
        fontName="NotoSans",
        fontSize=9.5,
        leading=13,
    )

    url_style = ParagraphStyle(
        "ReportURL",
        parent=styles["BodyText"],
        fontName="NotoSans",
        fontSize=7,
        leading=9,
        wordWrap="CJK",
    )
    detail_style = ParagraphStyle(
        "ReportDetail",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        wordWrap="CJK",
    )
    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=7,
        leading=9,
        wordWrap="CJK",
    )

    def display(value, default="N/A"):
        if value is None or pd.isna(value):
            return default
        value = str(value).strip()
        return value if value else default

    def para(value, style=detail_style):
        return Paragraph(escape(display(value)), style)

    story = [
        Paragraph("Competitor Analysis Report", styles["Title"]),
        Spacer(1, 4),

        Paragraph(
            f"<b>Business:</b> {escape(display(business, 'Not provided'))}",
            meta_style,
        ),
        Paragraph(
            f"<b>Industry:</b> {escape(display(industry, 'Not provided'))}",
            meta_style,
        ),
        Paragraph(
            f"<b>Location:</b> {escape(display(location, 'Not provided'))}",
            meta_style,
        ),
        Paragraph(
            f"<b>Generated:</b> {escape(display(timestamp, 'Unknown'))}",
            meta_style,
        ),

        Spacer(1, 6),

        Paragraph(
            "Competitive scores use available metrics only. Coverage is the "
            "percentage of configured metric weights represented by entered "
            "values. Low coverage means the comparison is incomplete; blank "
            "metrics mean unknown, not zero. Website-audit scores are separate "
            "heuristic technical checks, not Google rankings or overall "
            "competitor rankings.",
            intro_style,
        ),

        Spacer(1, 6),

        Paragraph("1. Competitive comparison", styles["Heading2"]),
        Spacer(1, 6),
    ]

    # Keep competitive metrics separate from technical website-audit scores.
    comparison_data = [[
        "Competitor",
        "Available-metric score",
        "Coverage",
        "Website audit score",
        "Audit status",
        "Website",
    ]]

    for _, row in results.iterrows():
        score = row.get("Overall score")
        score_text = (
            f"{float(score):.1f}" if pd.notna(score) else "N/A"
        )

        coverage = row.get("Coverage %")
        coverage_text = (
            f"{float(coverage):.0f}%" if pd.notna(coverage) else "N/A"
        )

        audit_score = row.get("Website audit score")
        audit_score_text = (
            f"{float(audit_score):.0f}/100"
            if pd.notna(audit_score)
            else "N/A"
        )

        comparison_data.append([
            para(row.get("Competitor", "")),
            score_text,
            coverage_text,
            audit_score_text,
            para(row.get("Audit status", "Not audited"), small_style),
            Paragraph(
                escape(display(row.get("Website", ""))),
                url_style,
            ),
        ])

    comparison_table = Table(
        comparison_data,
        colWidths=[115, 85, 50, 85, 70, 285],
        repeatRows=1,
    )
    comparison_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "NotoSans-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    story.extend([
        comparison_table,
        Spacer(1, 12),
        Paragraph("2. Website audit evidence", styles["Heading2"]),
        Spacer(1, 3),
        Paragraph(
            "OK means successfully fetched, not necessarily an active "
            "competitor website. Website quality is a heuristic score, "
            "not a search ranking. Review content warnings manually.",
            audit_intro_style,
        ),
        Spacer(1, 4),
    ])

    company_page_blocks = []

    for index, (_, row) in enumerate(results.iterrows(), start=1):
        competitor = display(row.get("Competitor", "Unnamed competitor"))
        website = display(row.get("Website", ""))
        warning = display(
            row.get("Content warning", ""),
            "None recorded",
        )
        audit_notes = display(row.get("Audit notes", ""))

        http_status = row.get("HTTP status")
        http_text = (
            str(int(http_status))
            if pd.notna(http_status)
            else "N/A"
        )

        https_value = row.get("HTTPS")
        https_text = display(https_value)

        audit_score = row.get("Website audit score")
        quality_text = (
            f"{float(audit_score):.0f}/100"
            if pd.notna(audit_score)
            else "N/A"
        )

        h1 = display(row.get("H1 count"))
        h2 = display(row.get("H2 count"))
        images = display(row.get("Images"))
        missing_alt = display(row.get("Images missing alt"))
        word_count = display(row.get("Word count"))

        evidence_rows = [
            ["Website", Paragraph(escape(website), url_style)],
            [
                "HTTP / HTTPS / Audit status",
                para(
                    f"{http_text} / {https_text} / "
                    f"{display(row.get('Audit status'), 'Not audited')}"
                ),
            ],
            ["Website quality", para(quality_text)],
            ["Page title", para(row.get("Page title", ""))],
            ["Meta description", para(row.get("Meta description", ""))],
            ["Heading counts (H1 / H2)", para(f"{h1} / {h2}")],
            [
                "Images / Missing alt text",
                para(f"{images} / {missing_alt}"),
            ],
            ["Visible word count", para(word_count)],
            ["Audit notes", para(audit_notes)],
            ["Content warning", para(warning)],
        ]

        evidence_table = Table(
            evidence_rows,
            colWidths=[145, 515],
        )
        evidence_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8EEF5")),
            ("FONTNAME", (0, 0), (0, -1), "NotoSans-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))

        company_page_blocks.extend([
            Paragraph(
                f"<u>{escape(competitor)}</u>",
                styles["Heading3"],
            ),
            Spacer(1, 2),
            evidence_table,
            Spacer(1, 8),
        ])

        if index % 2 == 0:
            story.append(KeepTogether(company_page_blocks))
            company_page_blocks = []

            if index < len(results):
                story.append(PageBreak())

    # Handle a remaining company if the number of records is odd.
    if company_page_blocks:
        story.append(KeepTogether(company_page_blocks))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()

st.markdown("""
<style>
/* Main application title */
.stApp h1 {
    font-size: 2.2rem !important;
    font-weight: 750 !important;
}

/* Numbered section headings and other section headings */
.stApp h2,
.stApp h3 {
    font-size: 1.4rem !important;
    font-weight: 650 !important;
}

/* Main explanatory text */
.stApp p,
.stApp li,
.stApp label {
    font-size: 1rem;
    line-height: 1.55;
}

/* Supporting captions: smaller, but still readable */
.stApp [data-testid="stCaptionContainer"] {
    font-size: 0.88rem;
}

/* Sidebar headings remain distinct */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)
st.title("Competitor Analysis Agent")
st.caption("Local-first MVP | Public website checks + user-verified competitor metrics")

with st.sidebar:
    st.header("Your business")
    business_name = st.text_input("Business name", "My Business")
    industry = st.text_input("Industry", "Local business")
    location = st.text_input("Business location", "")
    st.text_input("Your website URL (reference only)", "")
    st.divider()
    st.subheader("Scoring weights")
    for metric, weight in WEIGHTS.items():
        st.write(f"{metric}: **{weight}%**")
    st.caption("Unknown metrics are excluded from the score, not treated as zero.")


# RESEARCH EVIDENCE DASHBOARD
st.subheader("Research Evidence Dashboard")
st.caption(
    "Published competitor research loaded from the evidence register. "
    "These records are separate from live website audits and scoring."
)

evidence_path = Path(__file__).resolve().parent / (
    "BrandMonk_Competitor_Evidence_Register.xlsx"
)

try:
    research_df = pd.read_excel(
        evidence_path,
        sheet_name="Course Register",
        header=0,
        nrows=26,
        usecols="A:E",
    )
    required_columns = [
        "City", "Course", "Competitor",
        "Published fee / duration", "Official URL",
    ]
    if not all(column in research_df.columns for column in required_columns):
        st.error(
            "The evidence workbook columns do not match the expected layout. "
            "The research table could not be displayed."
        )
    else:
        research_df = research_df[required_columns].copy()
        research_df = research_df.dropna(subset=["Competitor"])
        for column in required_columns:
            research_df[column] = research_df[column].fillna("").astype(str)

        city_options = ["All"] + sorted({
            city.strip()
            for value in research_df["City"]
            for city in value.split("&")
            if city.strip()
        })
        course_options = ["All"] + sorted(
            research_df["Course"].unique().tolist()
        )

        filter_city, filter_course = st.columns(2)
        with filter_city:
            if st.session_state.get("city_filter_default_version") != "coimbatore_default_v1":
                st.session_state["research_city_filter"] = (
                    "Coimbatore" if "Coimbatore" in city_options else "All"
                )
                st.session_state["city_filter_default_version"] = "coimbatore_default_v1"

            selected_city = st.selectbox(
                "Filter by city",
                city_options,
                key="research_city_filter",
            )
        with filter_course:
            selected_course = st.selectbox(
                "Filter by course",
                course_options,
                key="research_course_filter",
            )

        filtered_research = research_df.copy()
        if selected_city != "All":
            filtered_research = filtered_research[
                filtered_research["City"].apply(
                    lambda value: selected_city in [
                        item.strip() for item in value.split("&")
                    ]
                )
            ]
        if selected_course != "All":
            filtered_research = filtered_research[
                filtered_research["Course"] == selected_course
            ]

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Research records", len(filtered_research))
        metric2.metric(
            "Competitor names",
            filtered_research["Competitor"].nunique(),
        )
        metric3.metric(
            "Courses represented",
            filtered_research["Course"].nunique(),
        )

        st.dataframe(
            filtered_research,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Official URL": st.column_config.LinkColumn(
                    "Official source",
                    display_text="Open source",
                ),
            },
        )
        st.caption(
            "Fees and durations retain the evidence register's wording. "
            "Entries marked for confirmation have not been treated as verified "
            "quotes. Displaying a source does not mean a live website audit "
            "has been run."
        )

        st.subheader("Research shortlist: 3 relevant providers per course")
        st.caption(
            "Providers are selected from the current research filters and "
            "evidence-register order. This is a relevance shortlist, not a "
            "verified performance ranking. Fewer than three may appear when "
            "the filtered evidence contains fewer providers."
        )

        if not filtered_research.empty:
            shortlist = (
                filtered_research[
                    ["Course", "City", "Competitor",
                     "Published fee / duration", "Official URL"]
                ]
                .drop_duplicates(
                    subset=["Course", "Competitor", "City"]
                )
                .groupby("Course", sort=True, group_keys=False)
                .head(3)
                .reset_index(drop=True)
            )

            st.dataframe(
                shortlist,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Official URL": st.column_config.LinkColumn(
                        "Official source",
                        display_text="Open source",
                    ),
                },
            )

            course_count = shortlist["Course"].nunique()
            provider_count = shortlist[
                ["Course", "Competitor"]
            ].drop_duplicates().shape[0]
            st.info(
                f"Shortlist contains {provider_count} course-provider "
                f"listings across {course_count} course categories in the "
                "current filter. Validate fees and durations before making "
                "business decisions."
            )
        else:
            st.info(
                "No research records match the selected city and course filters."
            )

except FileNotFoundError:
    st.error(
        "Evidence workbook not found. Keep "
        "'BrandMonk_Competitor_Evidence_Register.xlsx' beside app.py."
    )
except Exception as exc:
    st.error(
        "Could not load the research evidence: "
        f"{type(exc).__name__}: {exc}"
    )

st.divider()

st.subheader("1. Competitor list")
st.write(
    "Enter competitor names and websites. Enter scores only when supported by "
    "verified evidence. Leave unknown metrics blank. Website audit scores are "
    "technical checks and are not automatically used as competitive scores."
)


def build_competitor_data(city):
    source = research_df[["City", "Competitor", "Official URL"]].copy()
    source = source.dropna(subset=["Competitor"])
    source["City"] = source["City"].fillna("").astype(str)
    source["Competitor"] = source["Competitor"].fillna("").astype(str).str.strip()
    source["Website"] = source["Official URL"].fillna("").astype(str).str.strip()

    if city != "All":
        source = source[source["City"].apply(
            lambda value: city in [item.strip() for item in value.split("&")]
        )]

    data = source[source["Competitor"] != ""][["Competitor", "Website"]]
    data = data.drop_duplicates(subset=["Competitor", "Website"]).copy()
    for metric in METRIC_COLUMNS:
        data[metric] = None
    return data.reset_index(drop=True)


def preserve_competitor_scores(new_data, old_data):
    if old_data is None or not hasattr(old_data, "iterrows"):
        return new_data

    old = old_data.copy()
    for column in ("Competitor", "Website"):
        if column not in old.columns:
            old[column] = ""
    old["Competitor"] = old["Competitor"].fillna("").astype(str).str.strip()
    old["Website"] = old["Website"].fillna("").astype(str).str.strip()

    previous_rows = {}
    for _, row in old.iterrows():
        pair = (row["Competitor"], row["Website"])
        previous_rows[pair] = row

    for index, row in new_data.iterrows():
        pair = (
            str(row["Competitor"]).strip(),
            str(row["Website"]).strip(),
        )
        previous = previous_rows.get(pair)
        if previous is not None:
            for metric in METRIC_COLUMNS:
                if metric in previous.index:
                    new_data.at[index, metric] = previous[metric]

    return new_data


if "competitor_score_cache" not in st.session_state:
    st.session_state["competitor_score_cache"] = (
        load_competitor_score_cache()
    )

default_data = build_competitor_data(selected_city)

if "competitor_editor_generation" not in st.session_state:
    st.session_state.competitor_editor_generation = 0

if st.session_state.get("competitor_data_source_version") != "research_register_city_v2":
    previous_data = st.session_state.get("competitors")
    st.session_state.competitors = preserve_competitor_scores(
        default_data.copy(), previous_data
    )
    st.session_state.competitor_data_source_version = "research_register_city_v2"
    st.session_state.competitor_editor_generation += 1


st.session_state.competitors = apply_competitor_score_cache(
    st.session_state.competitors,
    st.session_state["competitor_score_cache"],
)

edited = st.data_editor(
    st.session_state.competitors,
    num_rows="dynamic",
    use_container_width=True,
    key=f"competitor_editor_research_v2_{st.session_state.competitor_editor_generation}",
    column_config={
        "Competitor": st.column_config.TextColumn(required=True),
        "Website": st.column_config.TextColumn(help="Enter a domain or full URL."),
        **{
            metric: st.column_config.NumberColumn(
                metric,
                min_value=0,
                max_value=100,
                step=1,
                help="Optional verified score. Leave blank if unknown.",
            )
            for metric in METRIC_COLUMNS
        },
    },
)

if st.button("Load selected city", key="load_selected_city"):

    # Nonblank values update the cache. Blank cells never erase saved scores.
    score_cache = st.session_state.setdefault("competitor_score_cache", {})

    for _, row in edited.iterrows():
        competitor = str(row.get("Competitor", "")).strip()
        website = str(row.get("Website", "")).strip()
        if not competitor:
            continue

        pair = (competitor, website)
        cached_metrics = score_cache.setdefault(pair, {})

        for metric in METRIC_COLUMNS:
            value = row.get(metric)
            if pd.isna(value) or value == "":
                continue
            try:
                cached_metrics[metric] = max(
                    0.0, min(100.0, float(value))
                )
            except (ValueError, TypeError):
                continue

    save_competitor_score_cache(score_cache)

    next_data = apply_competitor_score_cache(
        build_competitor_data(selected_city),
        score_cache,
    )
    st.session_state.competitors = next_data
    st.session_state.competitor_editor_generation += 1
    st.rerun()


confirm_clear_scores = st.checkbox(
    "I understand this permanently clears all saved metric scores",
    key="confirm_clear_saved_scores",
)

if st.button(
    "Clear all saved metric scores",
    key="clear_all_saved_metric_scores",
    disabled=not confirm_clear_scores,
):
    score_cache = st.session_state.setdefault(
        "competitor_score_cache", {}
    )
    score_cache.clear()
    save_competitor_score_cache(score_cache)
    st.session_state.competitors = build_competitor_data(selected_city)
    st.session_state.competitor_editor_generation += 1
    st.rerun()

if st.button("Run website audits and score", type="primary"):
    records = []
    audits = []
    for _, row in edited.iterrows():
        name = str(row.get("Competitor", "") or "").strip()
        if not name:
            continue

        url = normalize_url(row.get("Website", ""))
        record = {"Competitor": name, "Website": url}
        for metric in METRIC_COLUMNS:
            raw = row.get(metric)
            if pd.isna(raw) or raw == "":
                record[metric] = None
            else:
                try:
                    record[metric] = max(0, min(100, float(raw)))
                except (ValueError, TypeError):
                    record[metric] = None

        audit = audit_website(url)
        if url and audit["Audit status"] == "OK":
            # Keep the technical audit separate from verified competitive metrics.
            record["Website audit score"] = audit["Website quality"]

        record.update({
            "Audit status": audit["Audit status"],
            "HTTP status": audit["HTTP status"],
            "HTTPS": audit["HTTPS"],
            "Page title": audit["Page title"],
            "Meta description": audit["Meta description"],
            "H1 count": audit["H1 count"],
            "H2 count": audit["H2 count"],
            "Images": audit["Images"],
            "Images missing alt": audit["Images missing alt"],
            "Word count": audit["Word count"],
            "Audit notes": audit["Audit notes"],
            "Content warning": audit["Content warning"],
        })
        score, coverage = calculate_score(record)
        record["Overall score"] = score
        record["Coverage %"] = coverage
        records.append(record)
        audits.append(audit)

    st.session_state.results = pd.DataFrame(records)
    st.session_state.audits = pd.DataFrame(audits)
    st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# RESEARCH REPORT EXPORTS
st.subheader("Research report exports")
st.caption(
    "Exports published competitor research from the evidence register. "
    "These records are not website-audit scores or a verified ranking."
)

if "filtered_research" in locals():
    if not filtered_research.empty:
        research_csv_data = filtered_research.to_csv(index=False).encode("utf-8-sig")

        research_col_csv, research_col_pdf = st.columns(2)
        with research_col_csv:
            st.download_button(
                "Download research CSV",
                data=research_csv_data,
                file_name="competitor_research_evidence.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_research_csv",
            )

        with research_col_pdf:
            try:
                research_buffer = BytesIO()
                research_document = SimpleDocTemplate(
                    research_buffer,
                    pagesize=landscape(A4),
                    rightMargin=24,
                    leftMargin=24,
                    topMargin=24,
                    bottomMargin=24,
                )
                research_styles = getSampleStyleSheet()
                research_styles["Title"].fontName = "NotoSans-Bold"
                research_styles["BodyText"].fontName = "NotoSans"
                cell_style = ParagraphStyle(
                    "ResearchCell",
                    parent=research_styles["BodyText"],
                    fontName="NotoSans",
                    fontSize=6.5,
                    leading=8,
                )

                research_story = [
                    Paragraph(
                        "Competitor Research Evidence Report",
                        research_styles["Title"],
                    ),
                    Spacer(1, 8),
                    Paragraph(
                        "Source: BrandMonk Competitor Evidence Register. "
                        "Published fees and durations may require confirmation. "
                        "This report does not represent live website audits or scored rankings.",
                        research_styles["BodyText"],
                    ),
                    Spacer(1, 10),
                ]

                research_headers = [
                    "City", "Course", "Competitor",
                    "Published fee / duration", "Official URL",
                ]
                research_table_data = [[
                    Paragraph(escape(str(header)), cell_style)
                    for header in research_headers
                ]]

                for _, research_row in filtered_research.iterrows():
                    url = str(research_row.get("Official URL", "") or "").strip()
                    cells = []
                    for column in research_headers[:-1]:
                        value = str(research_row.get(column, "") or "")
                        cells.append(Paragraph(escape(value), cell_style))

                    if url.startswith(("https://", "http://")):
                        url_cell = Paragraph(
                            f'<link href="{escape(url)}">Open official source</link>',
                            cell_style,
                        )
                    else:
                        url_cell = Paragraph(escape(url or "Not provided"), cell_style)

                    cells.append(url_cell)
                    research_table_data.append(cells)

                research_table = Table(
                    research_table_data,
                    colWidths=[75, 85, 115, 220, 235],
                    repeatRows=1,
                )
                research_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "NotoSans-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#F1F5F9")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                research_story.append(research_table)
                research_document.build(research_story)
                research_buffer.seek(0)
                research_pdf_data = research_buffer.getvalue()

                st.download_button(
                    "Download research PDF",
                    data=research_pdf_data,
                    file_name="competitor_research_evidence.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_research_pdf",
                )
            except Exception as exc:
                st.error(
                    f"Research PDF generation failed: {type(exc).__name__}: {exc}"
                )
    else:
        st.info("No research records match the selected city/course filters.")
else:
    st.warning("Research data is unavailable; check the evidence workbook and dashboard.")

if "results" not in st.session_state:
    st.info("Enter competitors and click **Run website audits and score** to generate the dashboard.")
    st.stop()

results = st.session_state.results.copy()
if results.empty:
    st.warning("No competitors have been entered. Add at least one competitor and run the analysis.")
    st.stop()

st.subheader("2. Analysis dashboard")
st.caption(f"Last run: {st.session_state.get('last_run', 'Unknown')}")

valid_scores = results["Overall score"].dropna()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Competitors", len(results))
c2.metric("Websites audited", int((results["Audit status"] == "OK").sum()))
c3.metric("Average available score", f"{valid_scores.mean():.1f}/100" if not valid_scores.empty else "N/A")
c4.metric("Average metric coverage", f"{results['Coverage %'].mean():.1f}%")

if results["Coverage %"].lt(50).any():
    st.warning(
        "Low metric coverage: at least one competitor has less than 50% "
        "of configured metric weights represented. Scores are preliminary "
        "available-metric scores, not comprehensive competitor rankings. "
        "Enter verified metric scores before drawing competitive conclusions."
    )

display_columns = [
    "Competitor", "Website", "Overall score", "Coverage %",
    "Audit status", "HTTP status", "Website audit score",
]
dashboard_df = (
    results[[column for column in display_columns if column in results.columns]]
    .sort_values("Overall score", ascending=False, na_position="last")
    .rename(columns={"Overall score": "Available-metric score"})
)

dashboard_style = dashboard_df.style
for column in ["HTTP status"]:
    if column in dashboard_df.columns:
        dashboard_style = dashboard_style.set_properties(
            subset=[column], **{"text-align": "right"}
        )
for column in ["Audit status"]:
    if column in dashboard_df.columns:
        dashboard_style = dashboard_style.set_properties(
            subset=[column], **{"text-align": "center"}
        )

st.dataframe(
    dashboard_style,
    use_container_width=True,
    hide_index=True,
)

scored = results.dropna(subset=["Overall score"]).copy()

if not scored.empty:
    chart = px.bar(
        scored.sort_values("Overall score", ascending=False),
        x="Competitor",
        y="Overall score",
        color="Overall score",
        range_y=[0, 100],
        title="Available-metric score comparison (not a complete ranking)",
        hover_data=["Coverage %"],
    )
    st.plotly_chart(chart, use_container_width=True)
else:
    st.info("The score leaderboard will appear when at least one verified metric score is available.")

st.subheader("3. Metric comparison")

metric_data = results[["Competitor"] + METRIC_COLUMNS].melt(
    id_vars="Competitor", var_name="Metric", value_name="Score"
).dropna(subset=["Score"])

if not metric_data.empty:
    metric_chart = px.bar(
        metric_data,
        x="Metric",
        y="Score",
        color="Competitor",
        barmode="group",
        range_y=[0, 100],
        title="Entered scores by metric",
    )
    st.plotly_chart(metric_chart, use_container_width=True)
else:
    st.info("No metric scores have been entered yet. Add verified scores in section 1 to compare competitors.")

st.subheader("4. Initial recommendations")

if not scored.empty:
    best = scored.sort_values("Overall score", ascending=False).iloc[0]
    st.write(
        f"- **Highest available-metric score (not a definitive competitor winner):** "
        f"{best['Competitor']} ({best['Overall score']:.1f}/100; "
        f"coverage {best['Coverage %']:.0f}%)."
    )

    gaps = []
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(results[metric], errors="coerce").dropna()
        if not values.empty and values.mean() < 60:
            gaps.append((metric, values.mean()))

    if gaps:
        for metric, average in sorted(gaps, key=lambda item: item[1]):
            st.write(f"- **Investigate {metric}:** entered scores average {average:.1f}/100.")
    else:
        st.write(
            "- Enter verified scores for more metrics to identify meaningful gaps. "
            "Missing metrics are not evidence of poor performance."
        )
else:
    st.info(
        "Recommendations will appear after verified metric scores are entered. "
        "Blank metrics are unknown, not zero, and should not be treated as weaknesses."
    )
st.subheader("5. Website audit details")
audit_df = st.session_state.get("audits", pd.DataFrame())
if not audit_df.empty:
    audit_columns = [
        "Website", "Audit status", "HTTP status", "HTTPS", "Page title",
        "Meta description", "H1 count", "H2 count", "Images",
        "Images missing alt", "Word count", "Website quality",
        "Audit notes", "Content warning",
    ]
    audit_display = audit_df[
        [column for column in audit_columns if column in audit_df.columns]
    ]

    audit_style = audit_display.style
    for column in [
        "HTTP status", "H1 count", "H2 count", "Images",
        "Images missing alt", "Word count",
    ]:
        if column in audit_display.columns:
            audit_style = audit_style.set_properties(
                subset=[column], **{"text-align": "right"}
            )

    for column in ["Audit status", "HTTPS"]:
        if column in audit_display.columns:
            audit_style = audit_style.set_properties(
                subset=[column], **{"text-align": "center"}
            )

    st.dataframe(
        audit_style,
        use_container_width=True,
        hide_index=True,
    )

# Show technical website-audit scores separately from competitive scores.
audit_score_columns = {"Competitor", "Website audit score"}
if audit_score_columns.issubset(results.columns):

    audit_chart_data = results[
        ["Competitor", "Website audit score", "Website", "Audit status"]
    ].copy()

    audit_chart_data["Website audit score"] = pd.to_numeric(
        audit_chart_data["Website audit score"], errors="coerce"
    )
    audit_chart_data = audit_chart_data.dropna(
        subset=["Website audit score"]
    )

    # Normalize names for grouping without changing the original results.
    audit_chart_data["Competitor"] = (
        audit_chart_data["Competitor"].astype(str).str.strip()
    )
    audit_chart_data = audit_chart_data[
        audit_chart_data["Competitor"].ne("")
    ]

    # Display one bar per competitor, retaining the highest audited score.
    audit_chart_data = (
        audit_chart_data.groupby("Competitor", as_index=False, sort=False)
        .agg(
            **{
                "Website audit score": ("Website audit score", "max"),
                "Website": ("Website", "first"),
                "Audit status": ("Audit status", "first"),
            }
        )
    )

    # Sort by score descending, then competitor name A-Z (case-insensitive).
    audit_chart_data["_sort_name"] = (
        audit_chart_data["Competitor"].str.casefold()
    )
    audit_chart_data = (
        audit_chart_data.sort_values(
            ["Website audit score", "_sort_name"],
            ascending=[False, True],
            kind="stable",
        )
        .drop(columns="_sort_name")
        .reset_index(drop=True)
    )


    st.subheader("6. Website audit scores: highest to lowest")
    st.caption(
        "Sorted from highest to lowest. Axis starts at zero and ends at 100. "
        "These are technical website-audit scores, not overall competitor "
        "rankings. Only entries with an available audit score are shown."
    )

    if not audit_chart_data.empty:
        audit_chart_data = audit_chart_data.sort_values(
            by=["Website audit score", "Competitor"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

        audit_chart_data["Chart label"] = (
            audit_chart_data["Competitor"].astype(str)
        )

        audit_chart = px.bar(
            audit_chart_data,
            x="Chart label",
            y="Website audit score",
            range_y=[0, 100],
            hover_data=["Competitor"],
            title="Website audit scores (highest to lowest)",
            category_orders={
                "Chart label": audit_chart_data["Chart label"].tolist()
            },
        )

        audit_chart.update_traces(
            texttemplate="%{y:.0f}%",
            textposition="outside",
            cliponaxis=False,
        )

        audit_chart.update_xaxes(
            categoryorder="array",
            categoryarray=audit_chart_data["Chart label"].tolist(),
            title="Competitor",
            tickangle=-35,
        )
        audit_chart.update_yaxes(
            range=[0, 110],
            title="Website audit score",
        )

        st.plotly_chart(audit_chart, use_container_width=True)
    else:
        st.info(
            "Website-audit scores will appear after a successful audit run."
        )

    st.subheader("7. Evidence-based conclusion")
    if not audit_chart_data.empty:
        highest = audit_chart_data["Website audit score"].max()
        highest_names = audit_chart_data.loc[
            audit_chart_data["Website audit score"].eq(highest),
            "Competitor",
        ].drop_duplicates().tolist()
        st.write(
            f"The highest recorded website-audit score is {highest:.0f}/100 "
            f"for {', '.join(highest_names)}. This describes website-audit "
            "results only and does not establish the best course provider."
        )

    score_coverage = pd.to_numeric(
        results.get("Coverage %", pd.Series(dtype=float)),
        errors="coerce",
    )
    if score_coverage.empty or score_coverage.mean() < 50:
        st.warning(
            "Competitive evidence is incomplete. Collect comparable, "
            "verified metrics for search visibility, reputation, paid "
            "activity, keyword strategy, social presence and local presence "
            "before declaring category winners. Blank metrics mean unknown, "
            "not zero."
        )
    else:
        st.write(
            "Review metric coverage alongside competitive scores before "
            "drawing conclusions. Website-audit scores remain a separate measure."
        )
    st.divider()
    st.subheader("8. Download competitor analysis report")

    report_timestamp = st.session_state.get("last_run", "Unknown")
    report_pdf = build_pdf(
        business_name,
        industry,
        location,
        results,
        report_timestamp,
    )

    st.download_button(
        label="Download competitor analysis PDF",
        data=report_pdf,
        file_name="competitor_analysis_report.pdf",
        mime="application/pdf",
        key="download_competitor_analysis_pdf",
    )


