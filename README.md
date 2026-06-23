# GeoMind — Multi-Modal Geospatial Intelligence

A production-grade pipeline combining **Computer Vision + NLP Entity Extraction + TF-IDF RAG + LLM Fusion** for geological analysis.

## Pipeline Architecture

```
Drone/Core Image → Groq Vision LLM (llama-3.2-11b-vision) → Visual Findings
Geological PDF   → PDF Chunker (overlap=100) → TF-IDF RAG Retrieval → Context
                                             → NLP Entity Extraction (JSON)
                                                    ↓
                                         Multi-Modal Fusion (llama-3.3-70b)
                                                    ↓
                                    Structured Geotechnical Report
```

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Key Upgrades Over Baseline

| Feature | Baseline | GeoMind |
|---|---|---|
| CV Model | ResNet-18 via HuggingFace (generic ImageNet labels) | Groq Vision LLM (real geological interpretation) |
| RAG Retrieval | Naive set intersection | TF-IDF weighted scoring with chunk overlap |
| NLP Pipeline | None | Structured JSON entity extraction (rock types, depths, hazards, formations, boring metrics) |
| Fusion Prompt | Single stream | Three-stream synthesis with discrepancy flagging |
| UI | Basic output | Pipeline breadcrumb, entity chips, telemetry stats, animated loading |

## Tech Stack

- **FastAPI** — REST API + server-side rendering
- **Groq** — llama-3.2-11b-vision (CV) + llama-3.3-70b-versatile (NLP + Fusion)
- **pypdf** — PDF ingestion with overlapping chunking
- **Pillow** — Image preprocessing before base64 encoding
- **Jinja2 + Bootstrap 5** — Frontend templates
