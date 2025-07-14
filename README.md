# 🤖 AI Document Assistant

A powerful, modern AI-powered document analysis and chat application built with FastAPI, featuring Retrieval-Augmented Generation (RAG), web search capabilities, and a beautiful responsive UI.

![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--Turbo-412991?style=for-the-badge&logo=openai)
![Milvus](https://img.shields.io/badge/Milvus-Vector--DB-00D4AA?style=for-the-badge&logo=milvus)

## ✨ Features

### 🧠 **AI-Powered Document Analysis**
- **RAG (Retrieval-Augmented Generation)**: Upload documents and ask questions with AI-generated answers based on your content
- **Multi-format Support**: PDF, DOCX, and TXT file processing
- **Smart Chunking**: Intelligent document segmentation with overlap for better context
- **Semantic Search**: Find relevant information using vector embeddings

### 🌐 **Web Search Integration**
- **Person Search**: Automatically search the web for information about people
- **Real-time Information**: Get current information from multiple web sources
- **Hybrid Responses**: Combine document knowledge with web information

### 🎨 **Modern User Interface**
- **Responsive Design**: Beautiful, modern UI that works on all devices
- **Dark/Light Mode**: Toggle between themes with persistent preferences
- **Drag & Drop**: Intuitive file upload with visual feedback
- **Real-time Status**: Live system status indicators
- **Smooth Animations**: Professional animations and transitions

### 🔧 **Advanced Features**
- **Vector Database**: Milvus Managed (Zilliz Cloud) for scalable vector storage
- **OpenAI Integration**: GPT-3.5-Turbo for intelligent responses
- **RESTful API**: Clean, well-documented API endpoints
- **Error Handling**: Comprehensive error handling and user feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Milvus Managed (Zilliz Cloud) account

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd fastapi-vibe-coding
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

5. **Test OpenAI connection** (optional but recommended)
   ```bash
   python test_openai.py
   ```

6. **Update Milvus configuration** (in `main.py`)
   ```python
   MILVUS_URI = "your-milvus-uri"
   MILVUS_TOKEN = "your-milvus-token"
   COLLECTION_NAME = "your-collection-name"
   ```

7. **Run the application**
   ```bash
   python main.py
   ```

8. **Open your browser**
   Navigate to `http://localhost:8000` to access the beautiful web interface!

## 📚 API Documentation

### Core Endpoints

#### 🔍 **Ask Questions**
```http
POST /ask
Content-Type: application/json

{
  "message": "What is this document about?"
}
```
- **RAG + Web Search**: Combines document context with web information
- **Smart Person Detection**: Automatically searches web for person-related queries
- **AI Synthesis**: Uses GPT-3.5-Turbo for intelligent responses

#### 📄 **Upload Documents**
```http
POST /upload-document
Content-Type: multipart/form-data

file: [PDF/DOCX/TXT file]
```
- **Multi-format Support**: PDF, DOCX, TXT files
- **Automatic Processing**: Chunking, embedding, and storage
- **Real-time Feedback**: Upload progress and status updates

#### 🌐 **Web Search**
```http
POST /web-search
Content-Type: application/json

{
  "query": "search term"
}
```
- **Direct Web Search**: Search the web for any topic
- **Content Extraction**: Parse and extract relevant information
- **Formatted Results**: Clean, structured search results

#### ➕ **Add Text Manually**
```http
POST /add-document-string
Content-Type: application/json

{
  "text": "Your text content here...",
  "document_name": "Optional Name"
}
```

#### 📋 **System Status**
```http
GET /api/hello
GET /test-ai
GET /api/status
POST /api/reinitialize-openai
```
- **Health Checks**: API and AI system status
- **Testing**: Verify AI functionality
- **Status Monitoring**: Comprehensive system status
- **Reinitialization**: Reinitialize OpenAI client if needed

## 🏗️ Architecture

### **Frontend**
- **Modern UI**: Responsive design with CSS Grid and Flexbox
- **JavaScript**: Vanilla JS with modern ES6+ features
- **Font Awesome**: Professional icons throughout
- **Inter Font**: Clean, modern typography

### **Backend**
- **FastAPI**: High-performance Python web framework
- **Pydantic**: Data validation and serialization
- **Async/Await**: Non-blocking I/O operations
- **CORS**: Cross-origin resource sharing enabled

### **AI & Search**
- **OpenAI GPT-3.5-Turbo**: Natural language processing
- **OpenAI Embeddings**: Text vectorization (Ada-002)
- **Milvus Vector DB**: Scalable vector storage and search
- **Google Search**: Web information retrieval

### **Data Processing**
- **PyPDF2**: PDF text extraction
- **python-docx**: DOCX file processing
- **BeautifulSoup**: Web content parsing
- **Chunking Algorithm**: Intelligent text segmentation

## 🎯 Use Cases

### 📊 **Document Analysis**
- **Research Papers**: Extract key insights and answer questions
- **Reports**: Analyze business reports and generate summaries
- **Manuals**: Get quick answers from technical documentation
- **Books**: Interactive reading with AI-powered insights

### 👥 **Person Research**
- **Professional Profiles**: Get information about people
- **Background Checks**: Research individuals and their work
- **Networking**: Learn about contacts and connections
- **Biographical Data**: Historical and current information

### 🔍 **Information Discovery**
- **Web Search**: Find current information on any topic
- **Fact Checking**: Verify information from multiple sources
- **Research**: Gather information for projects and studies
- **Learning**: Interactive learning with AI assistance

## 🛠️ Configuration

### **Environment Variables**
```bash
OPENAI_API_KEY=sk-your-openai-api-key
```

### **Milvus Configuration**
```python
MILVUS_URI = "https://your-cluster.zillizcloud.com"
MILVUS_TOKEN = "your-milvus-token"
COLLECTION_NAME = "your-collection-name"
DIMENSION = 3270  # OpenAI Ada embedding dimension
```

### **Customization Options**
- **Chunk Size**: Adjust document chunking parameters
- **Search Results**: Configure number of web search results
- **AI Model**: Switch between different OpenAI models
- **UI Theme**: Customize colors and styling

## 📱 Screenshots

### **Main Interface**
- Modern chat interface with document upload
- Real-time status indicators
- Dark/light mode toggle
- Responsive design for all devices

### **Document Processing**
- Drag & drop file upload
- Progress indicators
- Success/error feedback
- Document information display

### **AI Chat**
- Message history with timestamps
- Typing indicators
- Rich text responses
- Context-aware answers

## 🔧 Troubleshooting

### **OpenAI Connection Issues**
1. **Test your API key**:
   ```bash
   python test_openai.py
   ```

2. **Check environment variable**:
   ```bash
   echo $OPENAI_API_KEY
   ```

3. **Common issues**:
   - **Invalid API key**: Ensure your key starts with `sk-` and is correct
   - **No credits**: Check your OpenAI account billing
   - **Rate limits**: Wait a moment and try again
   - **Network issues**: Check your internet connection

4. **Reinitialize OpenAI client**:
   ```bash
   curl -X POST http://localhost:8000/api/reinitialize-openai
   ```

### **Milvus Connection Issues**
1. **Check configuration** in `main.py`
2. **Verify credentials** are correct
3. **Ensure collection exists** or can be created
4. **Check network connectivity** to Zilliz Cloud

### **General Issues**
- **Port conflicts**: Change port in `main.py` if 8000 is busy
- **Dependencies**: Ensure all packages are installed correctly
- **File permissions**: Check write permissions for uploads

## 🔧 Development
```
fastapi-vibe-coding/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── static/
│   └── index.html      # Modern web interface
├── myenv/              # Virtual environment
└── README.md          # This file
```

### **Key Dependencies**
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **OpenAI**: AI model integration
- **PyMilvus**: Vector database client
- **BeautifulSoup**: Web scraping
- **Font Awesome**: Icons

### **Running in Development**
```bash
# Enable auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use the built-in runner
python main.py
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** for the excellent web framework
- **OpenAI** for powerful AI models
- **Milvus** for scalable vector storage
- **Font Awesome** for beautiful icons
- **Inter Font** for modern typography

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yaswanth009/fastapi-vibe-coding/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yaswanth009/fastapi-vibe-coding/discussions)
- **Email**: yash.bigdata@hotmail.com

---

<div align="center">

**Built with ❤️ using FastAPI, OpenAI, and modern web technologies**

[![Star on GitHub](https://img.shields.io/github/stars/yourusername/fastapi-vibe-coding?style=social)](https://github.com/yourusername/fastapi-vibe-coding)
[![Fork on GitHub](https://img.shields.io/github/forks/yourusername/fastapi-vibe-coding?style=social)](https://github.com/yourusername/fastapi-vibe-coding)

</div>
