#!/usr/bin/env python3
"""
Test script to check if ngrok is working correctly
"""

import subprocess
import time
import requests
import sys

def test_ngrok():
    """Test ngrok installation and functionality"""
    
    print("🧪 Testing ngrok installation and functionality")
    print("=" * 50)
    
    # Test 1: Check if ngrok is installed
    print("1. Checking ngrok installation...")
    try:
        result = subprocess.run(["ngrok", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ ngrok is installed: {result.stdout.strip()}")
        else:
            print("❌ ngrok is not installed")
            print("   Install with: brew install ngrok")
            return False
    except FileNotFoundError:
        print("❌ ngrok command not found")
        print("   Install with: brew install ngrok")
        return False
    
    # Test 2: Start a temporary ngrok tunnel
    print("\n2. Starting temporary ngrok tunnel on port 3000...")
    try:
        # Start ngrok process
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", "3000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("   Waiting for ngrok to start...")
        time.sleep(5)
        
        # Test 3: Get ngrok tunnel URL
        print("\n3. Getting ngrok tunnel information...")
        try:
            response = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=5)
            if response.status_code == 200:
                tunnels = response.json().get("tunnels", [])
                
                https_tunnel = None
                for tunnel in tunnels:
                    if tunnel.get("proto") == "https":
                        https_tunnel = tunnel.get("public_url")
                        break
                
                if https_tunnel:
                    print(f"✅ ngrok tunnel URL: {https_tunnel}")
                    print(f"   Webhook URL would be: {https_tunnel}/webhook")
                    
                    # Test 4: Test if the tunnel responds (optional)
                    print(f"\n4. Testing tunnel accessibility...")
                    try:
                        test_response = requests.get(https_tunnel, timeout=10)
                        if test_response.status_code == 502:
                            print("✅ Tunnel is working (502 expected since no server on port 3000)")
                        else:
                            print(f"✅ Tunnel responds with status: {test_response.status_code}")
                    except Exception as e:
                        print(f"⚠️ Tunnel test warning: {e}")
                        print("   This is normal if no server is running on port 3000")
                    
                    success = True
                else:
                    print("❌ No HTTPS tunnel found")
                    success = False
            else:
                print(f"❌ Could not get tunnel info: HTTP {response.status_code}")
                success = False
                
        except Exception as e:
            print(f"❌ Error getting tunnel info: {e}")
            success = False
        
        # Cleanup
        print("\n5. Cleaning up...")
        ngrok_process.terminate()
        ngrok_process.wait()
        print("✅ ngrok process terminated")
        
        return success
        
    except Exception as e:
        print(f"❌ Error starting ngrok: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 ngrok Test Suite")
    print("This will test if ngrok is properly installed and working")
    print()
    
    success = test_ngrok()
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    
    if success:
        print("✅ ngrok is working correctly!")
        print("\n💡 Next steps:")
        print("1. Add your Twilio credentials to .env file")
        print("2. Start the RAG server: ./run_simple.sh")
        print("3. The server will automatically start ngrok")
        print("4. Use the webhook URL for Twilio configuration")
    else:
        print("❌ ngrok is not working properly")
        print("\n🔧 Troubleshooting:")
        print("1. Install ngrok: brew install ngrok")
        print("2. Make sure no other ngrok processes are running")
        print("3. Check your internet connection")
        print("4. Try running: ngrok http 8000 manually")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)