#!/usr/bin/env python3
"""
Setup Validation Script
Checks if API keys and configuration are properly set
"""

import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message, status="info"):
    """Print colored status message"""
    if status == "success":
        print(f"{Colors.GREEN}✅ {message}{Colors.END}")
    elif status == "error":
        print(f"{Colors.RED}❌ {message}{Colors.END}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠️ {message}{Colors.END}")
    else:
        print(f"{Colors.BLUE}ℹ️ {message}{Colors.END}")

def check_env_file():
    """Check if .env file exists and has required keys"""
    print(f"\n{Colors.BOLD}🔧 Checking Environment Configuration{Colors.END}")
    
    env_file = Path(".env")
    if not env_file.exists():
        print_status(".env file not found", "error")
        print("   Run './run_simple.sh' or 'python setup.py' to create template")
        return False
    
    print_status(".env file exists", "success")
    
    # Load environment variables
    load_dotenv()
    
    # Check Groq API key
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "your_groq_api_key_here" or groq_key == "gsk_your_actual_groq_api_key_here_it_starts_with_gsk":
        print_status("GROQ_API_KEY not configured", "error")
        print("   Get your free API key from: https://console.groq.com/")
        return False
    elif not groq_key.startswith("gsk_"):
        print_status("GROQ_API_KEY format looks incorrect (should start with 'gsk_')", "warning")
        return False
    else:
        print_status("GROQ_API_KEY is configured", "success")
    
    # Check Twilio credentials (optional)
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    if twilio_sid and twilio_token and twilio_sid != "your_twilio_account_sid_here":
        if twilio_sid.startswith("AC") and len(twilio_sid) == 34:
            print_status("Twilio credentials are configured", "success")
        else:
            print_status("Twilio Account SID format looks incorrect", "warning")
    else:
        print_status("Twilio credentials not configured (WhatsApp will not work)", "warning")
        print("   Get free credentials from: https://www.twilio.com/try-twilio")
    
    return True

def test_groq_api():
    """Test if Groq API key works"""
    print(f"\n{Colors.BOLD}🤖 Testing Groq API Connection{Colors.END}")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print_status("No Groq API key to test", "error")
        return False
    
    try:
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": "Hello, respond with just 'API test successful'"}],
            "max_tokens": 10
        }
        
        print_status("Testing Groq API connection...", "info")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            print_status(f"Groq API test successful: {answer}", "success")
            return True
        else:
            print_status(f"Groq API test failed: {response.status_code} - {response.text}", "error")
            return False
            
    except Exception as e:
        print_status(f"Groq API test failed: {str(e)}", "error")
        return False

def test_twilio_credentials():
    """Test if Twilio credentials work"""
    print(f"\n{Colors.BOLD}📱 Testing Twilio Credentials{Colors.END}")
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    if not account_sid or not auth_token or account_sid == "your_twilio_account_sid_here":
        print_status("No Twilio credentials to test", "warning")
        return False
    
    try:
        from twilio.rest import Client
        
        print_status("Testing Twilio credentials...", "info")
        client = Client(account_sid, auth_token)
        
        # Test by getting account info
        account = client.api.accounts(account_sid).fetch()
        print_status(f"Twilio credentials valid - Account: {account.friendly_name}", "success")
        return True
        
    except ImportError:
        print_status("Twilio library not installed", "warning")
        print("   Run: pip install twilio")
        return False
    except Exception as e:
        print_status(f"Twilio credentials test failed: {str(e)}", "error")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print(f"\n{Colors.BOLD}📦 Checking Dependencies{Colors.END}")
    
    required_packages = [
        ("fastapi", "FastAPI web framework"),
        ("gradio", "Admin UI framework"),
        ("chromadb", "Vector database"),
        ("sentence_transformers", "Embedding models"),
        ("requests", "HTTP client"),
        ("python-dotenv", "Environment variables")
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_status(f"{package} - {description}", "success")
        except ImportError:
            print_status(f"{package} - {description} (MISSING)", "error")
            missing_packages.append(package)
    
    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "error")
        print("   Run: pip install -r requirements_simple.txt")
        return False
    
    return True

def check_directories():
    """Check if required directories exist"""
    print(f"\n{Colors.BOLD}📁 Checking Directory Structure{Colors.END}")
    
    required_dirs = ["data", "uploads", "cache"]
    
    for directory in required_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            print_status(f"{directory}/ directory exists", "success")
        else:
            print_status(f"Creating {directory}/ directory", "info")
            dir_path.mkdir(exist_ok=True)
    
    return True

def main():
    """Main validation function"""
    print(f"{Colors.BOLD}🔍 RAG Server Setup Validation{Colors.END}")
    print("=" * 50)
    
    checks = [
        ("Environment Configuration", check_env_file),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directories),
        ("Groq API", test_groq_api),
        ("Twilio Credentials", test_twilio_credentials)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print_status(f"{check_name} check failed: {str(e)}", "error")
            results[check_name] = False
    
    # Summary
    print(f"\n{Colors.BOLD}📋 Validation Summary{Colors.END}")
    print("=" * 50)
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    for check_name, result in results.items():
        status = "success" if result else "error"
        print_status(f"{check_name}: {'PASS' if result else 'FAIL'}", status)
    
    print(f"\n{Colors.BOLD}Results: {passed_checks}/{total_checks} checks passed{Colors.END}")
    
    if passed_checks >= 3:  # Core checks (env, deps, dirs)
        print_status("✨ Core setup is ready! You can start the server.", "success")
        print("   Run: ./run_simple.sh")
        
        if not results.get("Groq API", False):
            print_status("⚠️ Add Groq API key for full functionality", "warning")
        
        if not results.get("Twilio Credentials", False):
            print_status("⚠️ Add Twilio credentials for WhatsApp bot", "warning")
    
    else:
        print_status("❌ Setup is incomplete. Please fix the issues above.", "error")
        
        # Provide helpful next steps
        if not results.get("Environment Configuration", False):
            print("\n📝 Next steps:")
            print("1. Get Groq API key from: https://console.groq.com/")
            print("2. Edit .env file and add your API key")
            print("3. Run this validation script again")
    
    return passed_checks >= 3

if __name__ == "__main__":
    sys.exit(0 if main() else 1)