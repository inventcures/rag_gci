#!/usr/bin/env python3
"""
Test both text and audio responses via webhook
"""

import requests
import time

def test_text_and_audio_response():
    """Test that both text and audio responses are sent"""
    
    webhook_url = "http://localhost:8002/webhook"
    
    # Test data for a medical question
    test_data = {
        'From': 'whatsapp:+919876543210',
        'To': 'whatsapp:+14155238886',
        'Body': 'What is a neuroma?',  # Medical question that should be in corpus
        'MediaUrl0': '',
        'MediaContentType0': '',
        'NumMedia': '0'
    }
    
    print("🧪 Testing Both Text and Audio Response")
    print("=" * 50)
    print(f"📡 Webhook URL: {webhook_url}")
    print(f"❓ Question: {test_data['Body']}")
    print("\n🚀 Sending test webhook...")
    
    try:
        start_time = time.time()
        response = requests.post(webhook_url, data=test_data, timeout=60)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Response Status: {response.status_code}")
        print(f"⏱️ Processing time: {elapsed_time:.2f} seconds")
        print(f"📄 Response Content: {response.text}")
        
        if response.status_code == 200:
            print("\n✅ Webhook processed successfully!")
            print("\n💡 Expected sequence in server logs:")
            print("1. 📤 STEP 1: Sending text response...")
            print("2. ✅ Text message sent successfully!")
            print("3. ⏳ Waiting 2 seconds before sending audio...")
            print("4. 📤 STEP 2: Starting TTS synthesis...")
            print("5. ✅ Audio message sent successfully!")
            print("6. 📊 SUMMARY: Text sent: True, Audio sent: True")
        else:
            print(f"\n❌ Webhook failed with status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        print("\n💡 Make sure the server is running:")
        print("   ./run_simple.sh --port 8002")

def test_server_health():
    """Test if server is running"""
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is running")
            print(f"📊 WhatsApp: {data.get('whatsapp_bot', 'not configured')}")
            print(f"🌐 Ngrok URL: {data.get('ngrok_url', 'not available')}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Both Text and Audio Responses")
    print("=" * 50)
    
    if test_server_health():
        print("\n🚀 Running webhook test...")
        test_text_and_audio_response()
    else:
        print("\n💡 Please start the server first:")
        print("   ./run_simple.sh --port 8002")