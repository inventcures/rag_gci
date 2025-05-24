#!/usr/bin/env python3
"""
Script to automatically configure Twilio WhatsApp webhook URL
Run this after starting the server with ngrok
"""

import os
import requests
from twilio.rest import Client
from dotenv import load_dotenv

def setup_webhook():
    load_dotenv()
    
    # Get Twilio credentials
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    
    if not account_sid or not auth_token:
        print("❌ Twilio credentials not found in .env file")
        return
    
    print("🔧 Setting up Twilio WhatsApp webhook...")
    print(f"📋 Account SID: {account_sid[:10]}...")
    
    # Get the current ngrok URL
    try:
        # ngrok exposes a local API to get tunnel info
        response = requests.get('http://localhost:4040/api/tunnels')
        tunnels = response.json()['tunnels']
        
        ngrok_url = None
        for tunnel in tunnels:
            if tunnel['proto'] == 'https':
                ngrok_url = tunnel['public_url']
                break
                
        if not ngrok_url:
            print("❌ No ngrok tunnel found. Make sure ngrok is running.")
            print("💡 Start your server without --no-ngrok flag")
            return
            
    except Exception as e:
        print(f"❌ Could not get ngrok URL: {e}")
        print("💡 Manual setup required - see instructions below")
        return
    
    webhook_url = f"{ngrok_url}/webhook"
    print(f"🌐 Ngrok URL found: {ngrok_url}")
    print(f"🔗 Webhook URL: {webhook_url}")
    
    # Configure Twilio (for sandbox, this is manual via console)
    print("\n📱 MANUAL WEBHOOK SETUP REQUIRED:")
    print("=" * 50)
    print("1. Go to: https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn")
    print("2. In the 'Sandbox Configuration' section:")
    print(f"3. Set 'When a message comes in' to: {webhook_url}")
    print("4. Set HTTP method to: POST")
    print("5. Click 'Save Configuration'")
    print("")
    print(f"🎯 Your webhook URL: {webhook_url}")
    print("")
    print("📱 Then test by sending messages to: whatsapp:+14155238886")

if __name__ == "__main__":
    setup_webhook()