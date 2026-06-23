import os
import io
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from transformers import pipeline
from groq import Groq
import pypdf
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI(title="GeoMind Platform")
templates = Jinja2Templates(directory="templates")

# Initialize Groq Client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Initialize CV Pipeline (ResNet-18 for memory efficiency on Render)
print("Initializing Lightweight CV Engine...")
cv_extractor = pipeline("image-classification", model="microsoft/resnet-18")

# Initialize Persistent ChromaDB & Embedding Function for RAG
print("Initializing Vector Database Client...")
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "submitted": False})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_assets(
    request: Request,
    image: UploadFile = File(...),
    pdf_report: UploadFile = File(...)
):
    if not groq_client:
        return "Error: GROQ_API_KEY environment variable is missing on the server."

    # 1. COMPUTER VISION LAYER
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    raw_cv_predictions = cv_extractor(pil_image)
    
    cv_features = [
        {"feature": pred["label"].split(",")[0], "confidence": round(pred["score"] * 100, 2)}
        for pred in raw_cv_predictions[:3]
    ]

    # 2. VECTOR RAG PIPELINE
    pdf_bytes = await pdf_report.read()
    pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    
    pdf_chunks = []
    chunk_size = 500
    
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            for i in range(0, len(text), chunk_size):
                pdf_chunks.append({
                    "text": text[i:i+chunk_size],
                    "metadata": {"page": page_num + 1}
                })

    try:
        chroma_client.delete_collection("geo_documents")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name="geo_documents", 
        embedding_function=embedding_fn
    )
    
    collection.add(
        documents=[c["text"] for c in pdf_chunks],
        metadatas=[c["metadata"] for c in pdf_chunks],
        ids=[f"chunk_{idx}" for idx in range(len(pdf_chunks))]
    )
    
    geotechnical_queries = [
        "What rock formation, lithology types, and stratigraphy layers are present?",
        "What are the structural excavation drilling depths and boring metrics?",
        "What major structural geotechnical hazards, faults, or risks were discovered?"
    ]
    
    retrieved_context_blocks = []
    for q in geotechnical_queries:
        results = collection.query(query_texts=[q], n_results=1)
        if results and results["documents"][0]:
            retrieved_context_blocks.append(results["documents"][0][0])
            
    rag_context = "\n---\n".join(retrieved_context_blocks)

    # 3. MULTI-MODAL LLM FUSION ENGINE
    fusion_prompt = f"""
    You are the flagship Multi-Modal Intelligence Engine for GeomatikAI.
    Your objective is to combine computer vision visual outputs with semantically retrieved text entities to generate an engineering report.

    [VISUAL EXTRACTION EVIDENCE (From Drone/Core Sample Photo)]
    Top Predicted Lithological Elements: {cv_features}

    [TEXTUAL RETRIEVED CONTEXT (From Vector Database RAG over Field Report)]
    {rag_context}

    Generate a comprehensive Markdown report structured exactly as follows:
    ### 🔬 1. Cross-Verification Analysis
    (Synthesize if the visual data matches the report textual details.)
    
    ### 🚧 2. Identified Engineering Subsurface Risks
    (Detail explicit hazards, structural failures, or water boundaries mentioned in text or implied via vision.)

    ### 🛠️ 3. Decisive Actionable Site Recommendations
    (Provide concrete physical engineering guidance for site leads.)
    """

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": fusion_prompt}],
            model="llama-3.3-70b",
            temperature=0.2
        )
        fusion_report_md = response.choices[0].message.content
    except Exception as e:
        fusion_report_md = f"**Pipeline Error synthesizing data via Groq Engine:** {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "cv_features": cv_features,
        "fusion_report": fusion_report_md,
        "submitted": True
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
