#!/usr/bin/env python3
"""
Terminal Demo Video Generator
=============================

Generates a video showing actual terminal output from running
the realistic clinical unit test cases. This shows the REAL code
in action, not mockups.
"""

import os
import sys
import subprocess
from pathlib import Path
from moviepy import VideoClip, concatenate_videoclips
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Settings
WIDTH, HEIGHT = 1920, 1080
FPS = 30
LINE_HEIGHT = 22
CHAR_WIDTH = 11

def load_font(size):
    """Load monospace font for terminal look"""
    try:
        # Try to get a monospace font
        return ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", size)
    except:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
        except:
            return ImageFont.load_default()

# Terminal colors
BG_COLOR = (30, 30, 30)
HEADER_BG = (50, 50, 50)
TEXT_COLOR = (240, 240, 240)
GREEN = (80, 250, 123)
YELLOW = (241, 250, 140)
RED = (255, 85, 85)
CYAN = (139, 233, 253)
PINK = (255, 121, 198)

def parse_ansi(text):
    """Parse ANSI color codes and return styled segments"""
    # Simple ANSI parser - handles basic colors
    segments = []
    current_color = TEXT_COLOR
    current_text = ""
    
    i = 0
    while i < len(text):
        if text[i] == '\x1b' and i + 1 < len(text) and text[i + 1] == '[':
            # Found escape sequence
            if current_text:
                segments.append((current_text, current_color))
                current_text = ""
            
            # Find end of sequence
            j = i + 2
            while j < len(text) and text[j] not in 'm':
                j += 1
            
            code = text[i+2:j]
            
            # Parse color codes
            if code == '0' or code == '39' or code == '22' or code == '24':
                current_color = TEXT_COLOR
            elif code == '1':  # Bold
                pass  # We'll handle bold separately
            elif code == '92' or code == '32':  # Green
                current_color = GREEN
            elif code == '93' or code == '33':  # Yellow
                current_color = YELLOW
            elif code == '91' or code == '31':  # Red
                current_color = RED
            elif code == '94' or code == '34':  # Blue
                current_color = CYAN
            elif code == '95' or code == '35':  # Magenta/Pink
                current_color = PINK
            elif code == '96' or code == '36':  # Cyan
                current_color = CYAN
            
            i = j + 1
        else:
            current_text += text[i]
            i += 1
    
    if current_text:
        segments.append((current_text, current_color))
    
    return segments

def create_terminal_frame(lines, scroll_y=0, cursor_pos=None):
    """Create a terminal frame with given lines"""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = load_font(16)
    
    # Draw header bar
    draw.rectangle([0, 0, WIDTH, 30], fill=HEADER_BG)
    draw.text((10, 5), "palli-sahayak@terminal: ~/rag_gci", fill=TEXT_COLOR, font=font)
    draw.text((WIDTH - 200, 5), "python3 test_realistic_clinical_scenarios.py", fill=GREEN, font=font)
    
    # Draw terminal content
    y = 40 - scroll_y
    
    for i, line in enumerate(lines):
        if y > HEIGHT:
            break
        if y < 30:
            y += LINE_HEIGHT
            continue
        
        # Parse ANSI codes and draw segments
        segments = parse_ansi(line)
        x = 20
        
        for text, color in segments:
            draw.text((x, y), text, fill=color, font=font)
            x += len(text) * CHAR_WIDTH
        
        y += LINE_HEIGHT
    
    # Draw cursor if specified
    if cursor_pos:
        cx, cy = cursor_pos
        draw.rectangle([cx, cy, cx + 10, cy + 20], fill=GREEN)
    
    return np.array(img)

def read_terminal_output(filepath):
    """Read terminal output file"""
    with open(filepath, 'r') as f:
        return f.readlines()

def create_typing_animation(lines, duration_per_line=0.1):
    """Create typing animation for terminal output"""
    frames = []
    current_lines = []
    
    for line in lines:
        # Remove trailing newline
        line = line.rstrip('\n')
        
        # Type out the line character by character (faster for video)
        for i in range(1, len(line) + 1, 3):  # Skip by 3 chars for speed
            partial_line = line[:i]
            temp_lines = current_lines + [partial_line]
            
            # Calculate scroll if needed
            total_height = len(temp_lines) * LINE_HEIGHT + 50
            scroll_y = max(0, total_height - HEIGHT + 100)
            
            frame = create_terminal_frame(temp_lines, scroll_y)
            frames.append(frame)
        
        # Add complete line
        current_lines.append(line)
        
        # Calculate scroll
        total_height = len(current_lines) * LINE_HEIGHT + 50
        scroll_y = max(0, total_height - HEIGHT + 100)
        
        # Add a few frames for the complete line
        for _ in range(2):
            frame = create_terminal_frame(current_lines, scroll_y)
            frames.append(frame)
    
    return frames

def create_video_from_frames(frames, output_path, fps=30):
    """Create video from frames"""
    def make_frame(t):
        frame_idx = min(int(t * fps), len(frames) - 1)
        return frames[frame_idx]
    
    duration = len(frames) / fps
    clip = VideoClip(make_frame, duration=duration)
    
    clip.write_videofile(
        output_path,
        fps=fps,
        codec='libx264',
        audio=False,
        preset='medium'
    )
    
    return output_path

def main():
    """Generate terminal demo video"""
    print("=" * 70)
    print("🎬 Terminal Demo Video Generator")
    print("=" * 70)
    print()
    
    # First run the tests to generate fresh output
    print("Running clinical test scenarios...")
    result = subprocess.run(
        ['python3', 'test_realistic_clinical_scenarios.py'],
        capture_output=True,
        text=True,
        cwd='/Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci'
    )
    
    # Save output
    output_text = result.stdout + result.stderr
    lines = output_text.split('\n')
    
    print(f"Captured {len(lines)} lines of terminal output")
    print()
    
    # Create typing animation
    print("Generating typing animation...")
    frames = create_typing_animation(lines)
    
    print(f"Generated {len(frames)} frames")
    print()
    
    # Create video
    output_path = "palli_sahayak_terminal_demo.mp4"
    print(f"Creating video: {output_path}")
    
    create_video_from_frames(frames, output_path, fps=FPS)
    
    print()
    print("=" * 70)
    print(f"✅ Terminal demo video created: {output_path}")
    print(f"   Duration: {len(frames)/FPS:.1f} seconds")
    print(f"   Resolution: {WIDTH}x{HEIGHT}")
    print("=" * 70)

if __name__ == "__main__":
    main()
