from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, Union, List
import io
import PyPDF2
import docx
import os

import openai
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import uuid
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re

# Milvus Managed (Zilliz Cloud) credentials
MILVUS_URI = "***" # e.g., "https://in01-xxxx.api.gcp-us-west1.zillizcloud.com"
MILVUS_TOKEN = "***"

COLLECTION_NAME = "DataExpertLabTestYaswanth"
DIMENSION = 3270  # OpenAI Ada embedding dimension

# Read OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client properly
client = None
openai_status = "Not configured"

def initialize_openai_client():
    """Initialize and test OpenAI client."""
    global client, openai_status
    
    if not OPENAI_API_KEY:
        openai_status = "No API key found"
        print("❌ OPENAI_API_KEY environment variable not set")
        return False
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Test the connection with a simple API call
        test_response = client.models.list()
        openai_status = "Connected"
        print("✅ OpenAI client initialized and tested successfully")
        return True
        
    except Exception as e:
        openai_status = f"Connection failed: {str(e)}"
        print(f"❌ Error initializing OpenAI client: {e}")
        client = None
        return False

# Initialize on startup
initialize_openai_client()

app = FastAPI(
    title="FastAPI Vibe Coding, first program by Yaswanth",
    description="A simple FastAPI application for learning vibe coding with Cursor",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
document_content = ""
document_filename = ""

# Milvus configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "DataExpertLabTestYaswanth"
DIMENSION = 3270  # Dimension for sentence-transformers embeddings

# Initialize Milvus connection
def init_milvus():
    try:
        connections.connect(
            alias="default",
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            secure=True
        )
        print("✅ Connected to Milvus Managed (Zilliz Cloud) with token successfully")
        
        # Create collection if it doesn't exist
        if not utility.has_collection(COLLECTION_NAME):
            create_collection()
        else:
            print(f"✅ Collection '{COLLECTION_NAME}' already exists")
            
    except Exception as e:
        print(f"❌ Error connecting to Milvus Managed: {e}")

def create_collection():
    """Create Milvus collection for document chunks."""
    try:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        ]
        
        schema = CollectionSchema(fields=fields, description="Document chunks with embeddings")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        print(f"✅ Collection '{COLLECTION_NAME}' created successfully")
        
    except Exception as e:
        print(f"❌ Error creating collection: {e}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def get_embedding(text: str) -> list:
    """Get embedding for text using OpenAI's embedding API and pad to DIMENSION."""
    if not client:
        print("❌ OpenAI client not initialized. Check your OPENAI_API_KEY environment variable.")
        return []
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        emb = response.data[0].embedding
        if len(emb) < DIMENSION:
            emb = emb + [0.0] * (DIMENSION - len(emb))
        elif len(emb) > DIMENSION:
            emb = emb[:DIMENSION]
        return emb
    except Exception as e:
        print(f"❌ Error generating embedding with OpenAI: {e}")
        return []

def insert_document_chunks(document_text: str, document_name: str):
    """Insert document chunks into Milvus."""
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # Chunk the document
        chunks = chunk_text(document_text)
        
        # Prepare data for insertion (each field is a list)
        ids = []
        docnames = []
        chunk_texts = []
        embeddings = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_name}_{i}_{uuid.uuid4().hex[:8]}"
            embedding = get_embedding(chunk)
            if embedding and len(embedding) == DIMENSION:
                ids.append(chunk_id)
                docnames.append(document_name)
                chunk_texts.append(chunk)
                embeddings.append(embedding)
            else:
                print(f"❌ Skipping chunk {i}: embedding failure or dimension mismatch (got {len(embedding) if embedding else 0})")
        
        if ids:
            data = [ids, docnames, chunk_texts, embeddings]
            collection.insert(data)
            collection.flush()
            print(f"✅ Inserted {len(ids)} chunks for document '{document_name}'")
        else:
            print("❌ No valid chunks were inserted (embedding failure or dimension mismatch)")
            
    except Exception as e:
        print(f"❌ Error inserting document chunks: {e}")

def search_similar_chunks(query: str, top_k: int = 3) -> List[Dict]:
    """Search for similar chunks in Milvus."""
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        # Search for similar chunks
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["document_name", "chunk_text"]
        )
        
        similar_chunks = []
        for hits in results:
            for hit in hits:
                similar_chunks.append({
                    "document_name": hit.entity.get("document_name"),
                    "chunk_text": hit.entity.get("chunk_text"),
                    "score": hit.score
                })
        
        return similar_chunks
        
    except Exception as e:
        print(f"❌ Error searching similar chunks: {e}")
        return []


@app.get("/")
async def root() -> FileResponse:
    """Serve the index.html file."""
    return FileResponse("static/index.html")


@app.get("/api/hello")
async def hello_world() -> Dict[str, str]:
    """Hello world API endpoint."""
    return {"message": "Hello World!"}


@app.get("/hello/{name}")
async def hello_name(name: str) -> Dict[str, str]:
    """Personalized hello endpoint that takes a name parameter."""
    return {"message": f"Hello {name}!"}

@app.get("/test-ai")
async def test_ai() -> Dict[str, str]:
    """Test endpoint to verify AI is working."""
    if not client:
        return {
            "error": "OpenAI client not initialized", 
            "status": openai_status,
            "api_key_set": bool(OPENAI_API_KEY)
        }
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Give a brief, friendly response."},
                {"role": "user", "content": "Hello! Can you tell me what time it is?"}
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        return {
            "response": answer, 
            "status": "AI is working!",
            "openai_status": openai_status,
            "api_key_set": bool(OPENAI_API_KEY)
        }
    except Exception as e:
        return {
            "error": f"AI test failed: {str(e)}",
            "status": openai_status,
            "api_key_set": bool(OPENAI_API_KEY)
        }

@app.get("/api/status")
async def get_api_status() -> Dict[str, str]:
    """Get comprehensive API status including OpenAI connection."""
    return {
        "api_status": "Online",
        "openai_status": openai_status,
        "openai_connected": client is not None,
        "api_key_set": bool(OPENAI_API_KEY),
        "milvus_uri": MILVUS_URI if MILVUS_URI else "Not configured",
        "collection_name": COLLECTION_NAME
    }

@app.post("/api/reinitialize-openai")
async def reinitialize_openai() -> Dict[str, str]:
    """Reinitialize OpenAI client (useful if API key was updated)."""
    success = initialize_openai_client()
    return {
        "success": success,
        "status": openai_status,
        "message": "OpenAI client reinitialized successfully" if success else "Failed to reinitialize OpenAI client"
    }

@app.post("/debug-person-search")
async def debug_person_search(query: Dict[str, str]) -> Dict[str, str]:
    """Debug endpoint to test person search functionality."""
    user_message = query.get("query", "")
    if not user_message:
        return {"error": "No query provided"}
    
    # Check if this is a person-related query
    person_keywords = ["who is", "tell me about", "information about", "profile of", "background of"]
    is_person_query = any(keyword in user_message.lower() for keyword in person_keywords)
    
    # Test web search
    web_search_info = ""
    if is_person_query:
        print(f"🔍 Debug: Searching web for: {user_message}")
        web_search_info = extract_person_info(user_message)
    
    return {
        "query": user_message,
        "is_person_query": is_person_query,
        "web_search_info": web_search_info,
        "web_search_length": len(web_search_info) if web_search_info else 0
    }

@app.post("/web-search")
async def web_search(query: Dict[str, str]) -> Dict[str, str]:
    """Perform a web search and return results."""
    search_query = query.get("query", "")
    if not search_query:
        return {"error": "No search query provided"}
    
    try:
        print(f"🔍 Performing web search for: {search_query}")
        search_results = search_google(search_query, num_results=5)
        
        if not search_results:
            return {"response": f"No results found for '{search_query}'"}
        
        # Format results
        formatted_results = "Web search results:\n\n"
        for i, result in enumerate(search_results, 1):
            formatted_results += f"{i}. {result['url']}\n{result['content']}\n\n"
        
        return {"response": formatted_results}
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return {"error": f"Web search failed: {str(e)}"}


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting DOCX text: {str(e)}"

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file."""
    try:
        return file_content.decode('utf-8').strip()
    except Exception as e:
        return f"Error extracting TXT text: {str(e)}"

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Union[str, bool]]:
    """Upload and process a document, then add its chunks to Milvus."""
    global document_content, document_filename
    
    if not file.filename:
        return {"error": "No file provided"}
    
    # Check file type
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_extension = os.path.splitext(file.filename.lower())[1]
    
    if file_extension not in allowed_extensions:
        return {"error": f"Unsupported file type. Please upload PDF, DOCX, or TXT files."}
    
    try:
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        if file_extension == '.pdf':
            extracted_text = extract_text_from_pdf(content)
        elif file_extension == '.docx':
            extracted_text = extract_text_from_docx(content)
        elif file_extension == '.txt':
            extracted_text = extract_text_from_txt(content)
        else:
            return {"error": "Unsupported file type"}
        
        if extracted_text.startswith("Error"):
            return {"error": extracted_text}
        
        # Store document content and filename
        document_content = extracted_text
        document_filename = file.filename
        
        # --- Milvus Insertion Logic ---
        try:
            insert_document_chunks(extracted_text, file.filename)
            rag_status = "Document chunks and embeddings inserted into Milvus."
        except Exception as e:
            rag_status = f"Failed to insert into Milvus: {str(e)}"
        # --- End Milvus Insertion Logic ---
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' uploaded and processed!",
            "filename": file.filename,
            "content_preview": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
            "rag_status": rag_status
        }
        
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

@app.post("/ask")
async def ask_chatgpt(message: Dict[str, str]) -> Dict[str, str]:
    global document_content, document_filename

    user_message = message.get("message", "")
    if not user_message:
        return {"error": "No message provided"}

    # Check if this is a person-related query
    person_keywords = ["who is", "tell me about", "information about", "profile of", "background of"]
    is_person_query = any(keyword in user_message.lower() for keyword in person_keywords)
    
    # Retrieve relevant context from Milvus
    similar_chunks = search_similar_chunks(user_message, top_k=3)
    context = "\n\n".join([chunk["chunk_text"] for chunk in similar_chunks]) if similar_chunks else ""

    # If no document context and it's a person query, try Google search
    web_search_info = ""
    if not context and is_person_query:
        print(f"🔍 No document context found, searching web for: {user_message}")
        web_search_info = extract_person_info(user_message)
        if web_search_info and not web_search_info.startswith("I couldn't find"):
            context = f"Web search results:\n{web_search_info}"

    # Build prompt for ChatGPT
    if context:
        if web_search_info:
            system_prompt = (
                "You are a helpful AI assistant. Use the following web search information to answer the user's question accurately and comprehensively.\n\n"
                f"Web Search Information:\n{context}\n\n"
                "Provide a detailed, well-structured response based on the web search results. Include relevant facts, background information, and current details about the person."
            )
        else:
            system_prompt = (
                "You are a helpful AI assistant. Use the following document context to answer the user's question as accurately as possible.\n\n"
                f"Document context:\n{context}\n\n"
                "If the answer is not in the context, provide a helpful response based on your general knowledge."
            )
    else:
        # For person queries without context, be more helpful
        if is_person_query:
            system_prompt = (
                "You are a helpful AI assistant. The user is asking about a person, but I don't have specific information about them in my current context. "
                "Provide a helpful response based on your general knowledge, and suggest that they can ask me to search the web for more current information. "
                "Be informative and helpful rather than restrictive."
            )
        else:
            system_prompt = (
                "You are a helpful AI assistant. Answer the user's question to the best of your ability. "
                "If they ask about specific documents or files, remind them to upload a document first. "
                "For general questions, provide helpful and informative responses."
            )

    # Call OpenAI Chat API
    if not client:
        return {"error": "OpenAI client not initialized. Check your OPENAI_API_KEY environment variable."}
    
    try:
        print(f"🤖 Processing message: {user_message}")
        print(f"📄 Context found: {len(similar_chunks)} chunks")
        if web_search_info:
            print(f"🌐 Web search performed: {len(web_search_info)} characters")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        print(f"✅ AI Response: {answer[:100]}...")
        return {"response": answer}
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return {"error": f"OpenAI ChatGPT error: {str(e)}"}

@app.post("/add-document-string")
async def add_document_string(
    text: str = Body(..., embed=True),
    document_name: str = Body("manual_entry", embed=True)
) -> dict:
    """Add a string as a document, chunk, embed, and insert into Milvus for RAG."""
    if not text.strip():
        return {"error": "No text provided"}
    try:
        insert_document_chunks(text, document_name)
        return {"success": True, "message": f"Text added to RAG as document '{document_name}'"}
    except Exception as e:
        return {"error": f"Failed to add text to RAG: {str(e)}"}

def generate_rag_response(question: str, context: str, filename: str, similar_chunks: List[Dict]) -> str:
    """
    Generate a RAG response using the most relevant chunk(s) from Milvus.
    Returns the actual text of the most relevant chunk(s) as the answer.
    """
    if similar_chunks:
        # Return the most relevant chunk as the answer
        top_chunk = similar_chunks[0]
        answer = top_chunk["chunk_text"]
        score = top_chunk["score"]
        return (
            f"Answer based on the most relevant section from '{filename}':\n\n"
            f"{answer}\n\n"
            f"(Relevance score: {score:.2f})"
        )
    else:
        return (
            f"I couldn't find specific information related to your question in the document '{filename}'. "
            "Could you please rephrase or ask about a different aspect of the document?"
        )

def search_google(query: str, num_results: int = 3) -> list:
    """Search Google and return relevant information."""
    try:
        search_results = []
        for url in search(query, num_results=num_results, stop=num_results, pause=2.0):
            try:
                response = requests.get(url, timeout=5, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Extract relevant information (first 500 characters)
                    relevant_text = text[:500] + "..." if len(text) > 500 else text
                    search_results.append({
                        "url": url,
                        "content": relevant_text
                    })
            except Exception as e:
                print(f"❌ Error fetching content from {url}: {e}")
                continue
                
        return search_results
    except Exception as e:
        print(f"❌ Error in Google search: {e}")
        return []

def extract_person_info(query: str) -> str:
    """Extract information about a person using Google search."""
    try:
        # Clean the query to extract just the person's name
        person_name = query.replace("who is", "").replace("tell me about", "").replace("information about", "").replace("profile of", "").replace("background of", "").strip()
        
        # Search for the person with multiple queries for better results
        search_queries = [
            f"{person_name} profile biography career",
            f"{person_name} current information 2024",
            f"{person_name} background achievements"
        ]
        
        all_results = []
        for search_query in search_queries:
            search_results = search_google(search_query, num_results=2)
            all_results.extend(search_results)
        
        if not all_results:
            return f"I couldn't find information about {person_name} through web search."
        
        # Remove duplicates and combine results
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result['url'] not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result['url'])
        
        # Combine search results with better formatting
        combined_info = f"Web search results for '{person_name}':\n\n"
        for i, result in enumerate(unique_results[:4], 1):  # Limit to 4 best results
            combined_info += f"Source {i} ({result['url']}):\n{result['content']}\n\n"
        
        return combined_info
    except Exception as e:
        print(f"❌ Error extracting person info: {e}")
        return f"Error searching for information about {query}"

def generate_context_response(question: str, document_content: str, filename: str) -> str:
    """Generate a response based on document content and user question."""
    
    # Simple keyword-based response generation
    # In a real implementation, you would use OpenAI API here
    
    question_lower = question.lower()
    content_lower = document_content.lower()
    
    # Check if question contains common keywords
    if any(word in question_lower for word in ['summary', 'summarize', 'overview']):
        return f"Based on the document '{filename}', here's a summary: {document_content[:300]}..."
    
    elif any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where']):
        # Look for relevant content in the document
        relevant_sections = []
        sentences = document_content.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_lower.split()):
                relevant_sections.append(sentence.strip())
        
        if relevant_sections:
            return f"Based on the document '{filename}', here's what I found: {' '.join(relevant_sections[:2])}..."
        else:
            return f"I've reviewed the document '{filename}', but I couldn't find specific information related to your question. Could you please rephrase or ask about a different aspect of the document?"
    
    elif any(word in question_lower for word in ['find', 'search', 'locate']):
        return f"I've searched through the document '{filename}' for relevant information. Here's what I found: {document_content[:250]}..."
    
    else:
        return f"Based on the document '{filename}', I can help you with questions about its content. The document contains information about various topics. What specific aspect would you like to know more about?"

@app.get("/get-problem")
async def get_problem():
    """
    Return the contents of problem.txt as plain text.
    """
    try:
        with open("problem.txt", "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        return {"error": f"Could not read problem.txt: {str(e)}"}


@app.on_event("startup")
async def startup_event():
    """Initialize all connections on startup."""
    print("🚀 Starting AI Document Assistant...")
    
    # Initialize OpenAI
    openai_success = initialize_openai_client()
    
    # Initialize Milvus
    try:
        init_milvus()
        milvus_success = True
    except Exception as e:
        print(f"❌ Milvus initialization failed: {e}")
        milvus_success = False
    
    # Print startup summary
    print("\n📊 Startup Summary:")
    print(f"   OpenAI: {'✅ Connected' if openai_success else '❌ Failed'}")
    print(f"   Milvus: {'✅ Connected' if milvus_success else '❌ Failed'}")
    print(f"   API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Missing'}")
    print("🚀 Application ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
