# AgriAdvice: Offline RAG chatbot to address the digital divide

App for smallholder farmers needing climate-adapted advice without internet. Uses RAG with TinyLlama to pull answers from a document base, keeping outputs grounded.

Built through UN Women AI School, focusing on gaps like limited device access (e.g., shared computers in villages) and spotty connectivity.

## Background
Inspired by a classmate who suggested an offline AI that can run on flip phones. Scaled this idea to a technologically viable architecture. 
<img width="760" height="921" alt="offline_rag_architecture" src="https://github.com/user-attachments/assets/6f8e5f77-9ec3-4f94-a35a-105b1f436af3" />

## How It Works
Designed in two phases for low-connectivity areas.

**Setup (online, one-time)**:
- Extract text from 20-30 farming manuals using pypdf (text OCR) and pytesseract (image extraction). 
- Embed 20-30 PDFs using Hugging Face models.
- Store in ChromaDB vector DB.

**Offline use (laptops)**:
- small language model runs queries on embedded data. No internet needed.
- RAG setup: Model only uses retrieved docs to avoid hallucinations and false advice. 
- All open-source, no vendor costs.

Fallback idea: SMS for easily searchable information like weather/

## Tech Choices for Offline
- **TinyLlama**: Small enough for CPU on older laptops. 
- **RAG only**: Sticks to input data to prevent hallucinations.
- **Open source components only**: Removes financial barriers and prevents vendor lock-in. 
- English for MVP; will scale to multiple languages.
- Shared laptop target fits digital divide—avoids assuming personal phones.

## Setup
```
# Install
git clone seohyeonlee2020/offline-rag-chatbot.git
pip install -r requirements.txt

# Embed PDFs (online)
python utils/text_data_preprocessing.py 

# Run offline on localhost
streamlit run agriadvice_main.py  
```

## Next Steps
- Usage documentation
- Multilingual Quantize TinyLlama (4-bit via llama.cpp) to cut size/speed up inference.
- Group easily serchable information into a mass SMS service to reach users who do not have access to computers.



