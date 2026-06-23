import os
import io
import json
import base64
import math
from collections import Counter

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from groq import Groq
import pypdf

app = FastAPI(title="GeoMind Platform")
templates = Jinja2Templates(directory="templates")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ── TF-IDF RETRIEVAL (replaces naive set-intersection) ─────────────────────
def tfidf_retrieve(query: str, chunks: list, top_k: int = 2) -> list:
    """Score chunks using TF-IDF weighting instead of raw term overlap."""
    if not chunks:
        return []

    q_terms = set(query.lower().split())

    # IDF: inverse document frequency across all chunks
    doc_freq = Counter()
    for chunk in chunks:
        for word in set(chunk.lower().split()):
            doc_freq[word] += 1

    N = len(chunks)
    scores = []
    for chunk in chunks:
        words = chunk.lower().split()
        tf_counter = Counter(words)
        chunk_len = len(words) + 1
        score = 0.0
        for term in q_terms:
            tf = tf_counter.get(term, 0) / chunk_len
            idf = math.log((N + 1) / (doc_freq.get(term, 0) + 1))
            score += tf * idf
        scores.append(score)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    seen, results = set(), []
    for score, chunk in ranked:
        if chunk not in seen and score > 0:
            seen.add(chunk)
            results.append(chunk)
            if len(results) >= top_k:
                break
    return results


# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "submitted": False})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    pdf_report: UploadFile = File(...),
):
    if not groq_client:
        return HTMLResponse(
            "<h2 style='color:red;font-family:monospace'>Error: GROQ_API_KEY not set on server.</h2>",
            status_code=500,
        )

    # ── STEP 1: COMPUTER VISION via Groq Vision LLM ─────────────────────────
    # Key upgrade: llama-3.2-11b-vision-preview gives real geological
    # interpretation instead of generic ImageNet labels from ResNet.
    image_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img.thumbnail((1024, 1024))                     # resize before upload
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    vision_analysis = ""
    try:
        vr = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": (
                        "You are a geotechnical computer vision system. Analyze this image and extract:\n"
                        "1. Rock / soil type and lithology (be specific: sandstone, granite, clay, shale…)\n"
                        "2. Visible structural features (fractures, faults, bedding planes, joints, foliation)\n"
                        "3. Texture, grain size, and colour observations\n"
                        "4. Any visible hazard indicators (unstable zones, water seepage, weathering grade)\n"
                        "5. Estimated geological formation or era if determinable\n"
                        "Be concise and technically precise. Format as numbered bullet points."
                    )},
                ],
            }],
            temperature=0.1,
            max_tokens=500,
        )
        vision_analysis = vr.choices[0].message.content
    except Exception as e:
        vision_analysis = f"Vision analysis unavailable: {e}"

    # ── STEP 2: PDF INGESTION WITH OVERLAPPING CHUNKS ───────────────────────
    pdf_bytes = await pdf_report.read()
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    page_count = len(reader.pages)

    full_text = ""
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            full_text += txt + "\n"

    # Overlapping chunks improve retrieval boundary coverage
    chunk_size, overlap = 600, 100
    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunk = full_text[i : i + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)

    # ── STEP 3: NLP ENTITY EXTRACTION ───────────────────────────────────────
    # Separate Groq call that returns structured JSON — directly maps to
    # GeomatikAI's "parse, classify, entity extraction on geological reports".
    sample = full_text[:3000] if len(full_text) > 3000 else full_text
    entities: dict = {}
    try:
        er = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a geological NLP entity extraction engine. "
                        "Return ONLY valid JSON — no markdown fences, no preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Extract entities from the geological report below.\n"
                        "Return a JSON object with EXACTLY these keys:\n"
                        "  rock_types        (list of strings)\n"
                        "  formations        (list of strings)\n"
                        "  depths            (list of strings with units, e.g. '12.5 m')\n"
                        "  hazards           (list of strings)\n"
                        "  coordinates       (list of strings, empty if none)\n"
                        "  boring_metrics    (list of strings, e.g. 'SPT N=15 at 3m')\n"
                        "  recommendations   (list of strings)\n\n"
                        f"Report:\n{sample}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=700,
        )
        raw = er.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        entities = json.loads(raw)
    except Exception:
        entities = {
            "rock_types": ["Extraction failed — check PDF content"],
            "formations": [], "depths": [], "hazards": [],
            "coordinates": [], "boring_metrics": [], "recommendations": [],
        }

    # ── STEP 4: TF-IDF RAG RETRIEVAL ────────────────────────────────────────
    geo_queries = [
        "lithology stratigraphy rock formation type classification layer",
        "drilling depth boring SPT N-value excavation core sample measurement",
        "hazard fault fracture instability groundwater seepage risk assessment slope",
    ]
    retrieved = []
    for q in geo_queries:
        retrieved.extend(tfidf_retrieve(q, chunks, top_k=1))

    # Deduplicate while preserving order
    seen_r: set = set()
    unique_retrieved = [r for r in retrieved if not (r in seen_r or seen_r.add(r))]  # type: ignore[func-returns-value]
    rag_context = "\n---\n".join(unique_retrieved) or "No relevant segments retrieved."

    # ── STEP 5: MULTI-MODAL FUSION REPORT ───────────────────────────────────
    fusion_prompt = f"""You are GeomatikAI's Multi-Modal Geological Intelligence Engine.

Synthesize the three data streams below into a structured geotechnical evaluation report.
Cross-reference visual findings with documented metrics and flag discrepancies.

[STREAM 1 — COMPUTER VISION (Groq Vision LLM)]
{vision_analysis}

[STREAM 2 — NLP ENTITY EXTRACTION]
Rock Types      : {entities.get('rock_types', [])}
Formations      : {entities.get('formations', [])}
Depths          : {entities.get('depths', [])}
Hazards         : {entities.get('hazards', [])}
Boring Metrics  : {entities.get('boring_metrics', [])}
Recommendations : {entities.get('recommendations', [])}

[STREAM 3 — TF-IDF RAG RETRIEVED CONTEXT]
{rag_context}

Write a professional report with these exact sections:
## Executive Summary
## Visual–Textual Concordance Analysis
## Lithological Profile
## Structural Risk Assessment
## Geotechnical Recommendations

Be technically precise. Flag any discrepancies between visual evidence and documented data."""

    fusion_report = ""
    try:
        fr = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": fusion_prompt}],
            temperature=0.2,
            max_tokens=1200,
        )
        fusion_report = fr.choices[0].message.content
    except Exception as e:
        fusion_report = f"**Fusion pipeline error:** {e}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "submitted": True,
        "vision_analysis": vision_analysis,
        "entities": entities,
        "fusion_report": fusion_report,
        "pdf_stats": {
            "pages": page_count,
            "chunks": len(chunks),
            "retrieved": len(unique_retrieved),
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
