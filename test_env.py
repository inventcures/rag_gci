#!/usr/bin/env python3
"""
Test script to debug .env file loading
"""

import os
import sys
from pathlib import Path

def test_env_loading():
    """Test if environment variables are loaded correctly"""
    
    print("🧪 Testing .env file loading")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    print(f"1. .env file exists: {'✅ Yes' if env_file.exists() else '❌ No'}")
    
    if env_file.exists():
        print(f"   .env file path: {env_file.absolute()}")
        print(f"   .env file size: {env_file.stat().st_size} bytes")
        
        # Read and display .env content (masking sensitive data)
        print("\n2. .env file content:")
        with open(env_file, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Mask sensitive values
                        if 'API_KEY' in key or 'TOKEN' in key:
                            masked_value = value[:10] + '*' * (len(value) - 10) if len(value) > 10 else '*' * len(value)
                            print(f"   Line {i}: {key}={masked_value}")
                        else:
                            print(f"   Line {i}: {key}={value}")
                    else:
                        print(f"   Line {i}: {line}")
    
    # Test loading with python-dotenv
    print("\n3. Testing python-dotenv loading...")
    try:
        from dotenv import load_dotenv
        result = load_dotenv()
        print(f"   load_dotenv() result: {'✅ Success' if result else '❌ Failed'}")
    except ImportError:
        print("   ❌ python-dotenv not installed")
        print("   Install with: pip install python-dotenv")
        return False
    
    # Check environment variables
    print("\n4. Environment variables after loading:")
    
    # Test variables
    test_vars = {
        "GROQ_API_KEY": "Groq API Key",
        "TWILIO_ACCOUNT_SID": "Twilio Account SID", 
        "TWILIO_AUTH_TOKEN": "Twilio Auth Token",
        "TWILIO_WHATSAPP_FROM": "Twilio WhatsApp From"
    }
    
    all_configured = True
    
    for var_name, description in test_vars.items():
        value = os.getenv(var_name)
        if value:
            # Mask sensitive data
            if 'API_KEY' in var_name or 'TOKEN' in var_name:
                masked_value = value[:10] + '*' * (len(value) - 10) if len(value) > 10 else '*' * len(value)
                print(f"   ✅ {var_name}: {masked_value}")
            else:
                print(f"   ✅ {var_name}: {value}")
        else:
            print(f"   ❌ {var_name}: Not set")
            all_configured = False
    
    # Test Twilio configuration specifically
    print("\n5. Twilio configuration check:")
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    if twilio_sid and twilio_token and twilio_sid != "your_twilio_account_sid_here":
        print("   ✅ Twilio configuration: Valid")
        
        # Test Twilio connection
        print("\n6. Testing Twilio connection...")
        try:
            from twilio.rest import Client
            client = Client(twilio_sid, twilio_token)
            account = client.api.accounts(twilio_sid).fetch()
            print(f"   ✅ Twilio connection successful: {account.friendly_name}")
        except ImportError:
            print("   ⚠️ Twilio library not installed")
        except Exception as e:
            print(f"   ❌ Twilio connection failed: {e}")
    else:
        print("   ❌ Twilio configuration: Invalid or missing")
        if not twilio_sid:
            print("      Missing TWILIO_ACCOUNT_SID")
        if not twilio_token:
            print("      Missing TWILIO_AUTH_TOKEN")
        if twilio_sid == "your_twilio_account_sid_here":
            print("      TWILIO_ACCOUNT_SID is still placeholder value")
    
    return all_configured

def main():
    """Main test function"""
    print("🚀 Environment Variables Test")
    print("This will check if .env file is loaded correctly")
    print()
    
    success = test_env_loading()
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    
    if success:
        print("✅ All environment variables are configured correctly!")
        print("\n💡 Your .env configuration is working.")
        print("If you're still seeing 'not configured' messages,")
        print("the issue might be in how the main application loads the .env file.")
    else:
        print("❌ Some environment variables are missing or incorrect")
        print("\n🔧 Next steps:")
        print("1. Check your .env file content")
        print("2. Make sure there are no extra spaces around the = sign")
        print("3. Make sure there are no quotes around the values")
        print("4. Restart your server after making changes")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)