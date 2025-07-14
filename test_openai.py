#!/usr/bin/env python3
"""
OpenAI API Key Test Script
This script helps you verify if your OpenAI API key is working correctly.
"""

import os
import openai
import sys

def test_openai_connection():
    """Test OpenAI API key connection."""
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("\nTo set it, run:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    print(f"🔑 API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Initialize client
        client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")
        
        # Test models endpoint
        print("🔍 Testing models endpoint...")
        models = client.models.list()
        print(f"✅ Models endpoint working - Found {len(models.data)} models")
        
        # Test chat completion
        print("🤖 Testing chat completion...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello, OpenAI is working!'"}
            ],
            max_tokens=50
        )
        
        answer = response.choices[0].message.content
        print(f"✅ Chat completion working - Response: {answer}")
        
        # Test embeddings
        print("📊 Testing embeddings...")
        embedding_response = client.embeddings.create(
            input="Test text for embedding",
            model="text-embedding-ada-002"
        )
        
        embedding = embedding_response.data[0].embedding
        print(f"✅ Embeddings working - Vector dimension: {len(embedding)}")
        
        print("\n🎉 All tests passed! Your OpenAI API key is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        if "authentication" in str(e).lower() or "invalid" in str(e).lower():
            print("\n🔧 This looks like an authentication issue. Please check:")
            print("1. Your API key is correct")
            print("2. Your OpenAI account has credits")
            print("3. The API key has the necessary permissions")
        
        elif "rate limit" in str(e).lower():
            print("\n⏰ Rate limit exceeded. Please wait a moment and try again.")
        
        elif "quota" in str(e).lower():
            print("\n💰 Quota exceeded. Please check your OpenAI account billing.")
        
        return False

def main():
    """Main function."""
    print("🤖 OpenAI API Key Test Script")
    print("=" * 40)
    
    success = test_openai_connection()
    
    if success:
        print("\n✅ You can now run your FastAPI application!")
        print("   python main.py")
    else:
        print("\n❌ Please fix the issues above before running your application.")
        sys.exit(1)

if __name__ == "__main__":
    main() 