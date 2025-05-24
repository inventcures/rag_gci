#!/usr/bin/env python3
"""
Test script to simulate a WhatsApp webhook call with audio
This helps debug the audio processing pipeline
"""

import requests
import sys

def test_audio_webhook():
    """Test the webhook with a simulated audio message"""
    
    # Test URL - adjust port if needed
    webhook_url = "http://localhost:8002/webhook"
    
    # Simulate audio webhook data from Twilio
    test_data = {
        'From': 'whatsapp:+919876543210',  # Test number
        'To': 'whatsapp:+14155238886',     # Twilio sandbox number
        'Body': '',                        # Empty for audio messages
        'MediaUrl0': 'https://api.twilio.com/2010-04-01/Accounts/ACef0890e20d42b94a3a795c33a05fda95/Messages/MM123456789/Media/ME123456789',  # Fake URL for testing
        'MediaContentType0': 'audio/ogg; codecs=opus',
        'NumMedia': '1'
    }
    
    print("🧪 Testing WhatsApp Audio Webhook")
    print("=" * 50)
    print(f"📡 Webhook URL: {webhook_url}")
    print(f"📋 Test data: {test_data}")
    print("\n🚀 Sending test webhook...")
    
    try:
        response = requests.post(webhook_url, data=test_data, timeout=30)
        print(f"✅ Response Status: {response.status_code}")
        print(f"📄 Response Content: {response.text}")
        
        if response.status_code == 200:
            print("\n✅ Webhook processed successfully!")
            print("💡 Check the server logs for detailed audio processing information")
        else:
            print(f"\n❌ Webhook failed with status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        print("\n💡 Make sure the server is running on port 8002")

if __name__ == "__main__":
    test_audio_webhook()