# AgriAdvice: Offline RAG chatbot to address the digital divide

App for smallholder farmers needing climate-adapted advice without internet. Scaled ambitious idea from a classmate ("AI on flip phone offline") to a technologically viable architecture. Uses RAG with TinyLlama to pull answers from a document base, keeping outputs grounded. Focused on optimizing architecture for extremely low-resource contexts. Focused on gaps like limited device access (e.g., shared computers in villages) and spotty connectivity.

<img width="400" height="auto" alt="offline_rag_architecture" src="https://github.com/user-attachments/assets/6f8e5f77-9ec3-4f94-a35a-105b1f436af3" />

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

## Tech Choices for Offline Architecture
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
- Multilingual support
- Group easily serchable information into a mass SMS service to reach users who do not have access to computers.



