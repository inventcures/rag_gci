#!/usr/bin/env python3
"""
Palli Sahayak - System Demo Video (Screen Recording Style)
===========================================================

Generates a realistic screen-capture style video showing:
1. Real WhatsApp chat interface with Evidence Badges
2. Actual medication reminder setup for elderly patient
3. Live SIP-REFER handoff via Vobiz.ai integration

Resolution: 1920x1080 (Full HD)
Style: Realistic UI mockups with actual system behavior
"""

import os
from moviepy import VideoClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Settings
WIDTH, HEIGHT = 1920, 1080
FPS = 30

# Colors - Realistic UI palette
BG_DARK = (28, 28, 30)
WHATSAPP_BG = (11, 20, 26)
WHATSAPP_GREEN = (0, 92, 75)
WHATSAPP_BUBBLE = (0, 65, 54)
ACCENT_BLUE = (53, 120, 229)
ACCENT_GREEN = (52, 168, 83)
ACCENT_ORANGE = (251, 140, 0)
ACCENT_RED = (234, 67, 53)
ACCENT_GOLD = (245, 193, 108)
WHITE = (255, 255, 255)
GRAY_LIGHT = (220, 220, 220)
GRAY_MID = (150, 150, 150)
GRAY_DARK = (80, 80, 80)

def load_font(size, bold=False):
    """Load appropriate font"""
    try:
        if bold:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()

def add_rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    """Draw rounded rectangle"""
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

def add_text_with_shadow(draw, text, pos, font, fill, shadow_offset=(2, 2)):
    """Add text with shadow effect"""
    x, y = pos
    # Shadow
    draw.text((x + shadow_offset[0], y + shadow_offset[1]), text, font=font, fill=(0, 0, 0, 128))
    # Text
    draw.text((x, y), text, font=font, fill=fill)

def create_whatsapp_ui(draw, width, height):
    """Create WhatsApp web interface background"""
    # Main background
    draw.rectangle([0, 0, width, height], fill=WHATSAPP_BG)
    
    # Header
    draw.rectangle([0, 0, width, 65], fill=WHATSAPP_GREEN)
    
    # Profile icon (left)
    draw.ellipse([20, 10, 55, 45], fill=(200, 200, 200))
    
    # Header text
    font = load_font(18)
    draw.text((70, 15), "Palli Sahayak Helpline", fill=WHITE, font=load_font(20, True))
    draw.text((70, 40), "+91-XXXX-NH-HELP", fill=(180, 210, 200), font=font)
    
    # Header icons (right)
    icons_x = width - 150
    for icon in ["🔍", "⋮", "📎"]:
        draw.text((icons_x, 20), icon, fill=WHITE, font=load_font(20))
        icons_x += 40
    
    # Chat background pattern (subtle)
    for i in range(0, width, 100):
        for j in range(80, height, 100):
            draw.rectangle([i, j, i+50, j+50], fill=(15, 25, 30))
    
    return 80  # Return content start Y

def scene_evidence_badges(duration=12):
    """Scene 1: Real WhatsApp chat with Evidence Badges"""
    def make_frame(t):
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_DARK)
        draw = ImageDraw.Draw(img)
        
        # Create WhatsApp UI
        content_y = create_whatsapp_ui(draw, WIDTH, HEIGHT)
        
        # Timestamp
        font_small = load_font(12)
        draw.text((WIDTH//2, content_y + 10), "TODAY", fill=GRAY_MID, font=font_small, anchor="mm")
        
        # === USER MESSAGE (appears at t=0.5) ===
        if t > 0.5:
            # User bubble (right side)
            user_y = content_y + 40
            bubble_w = 500
            bubble_x = WIDTH - bubble_w - 80
            
            # Typing indicator first
            if t < 1.5:
                typing_progress = min(1.0, (t - 0.5) / 0.5)
                typing_text = "My mother has severe pain after chemo..."[:int(45 * typing_progress)]
            else:
                typing_text = "My mother has severe pain after chemo.\nWhat should I give her?"
            
            add_rounded_rect(draw, [bubble_x, user_y, bubble_x + bubble_w, user_y + 70], 
                           15, fill=(0, 130, 110))
            draw.text((bubble_x + 15, user_y + 12), typing_text, fill=WHITE, font=load_font(16))
            
            # Time
            draw.text((bubble_x + bubble_w - 50, user_y + 50), "10:32", fill=(180, 220, 200), font=font_small)
            draw.text((bubble_x + bubble_w - 25, user_y + 48), "✓✓", fill=(100, 200, 255), font=font_small)
        
        # === AI TYPING INDICATOR (t=2) ===
        if t > 2:
            ai_y = content_y + 130
            # Typing dots animation
            dots = min(3, int((t - 2) * 2) % 4)
            dot_text = "●" * dots + "○" * (3 - dots)
            add_rounded_rect(draw, [80, ai_y, 200, ai_y + 50], 20, fill=(50, 60, 65))
            draw.text((110, ai_y + 15), dot_text, fill=GRAY_MID, font=load_font(20))
        
        # === AI RESPONSE (t=3) ===
        if t > 3:
            ai_y = content_y + 130
            response_lines = [
                "Based on Max Healthcare case vignettes:",
                "",
                "1. Morphine 5-10mg oral every 4h PRN",
                "2. Monitor for constipation, drowsiness", 
                "3. Consult physician if pain persists >24h",
                "",
                "This matches 23 similar cases in our DB."
            ]
            
            # Reveal lines progressively
            lines_to_show = min(len(response_lines), int((t - 3) * 3))
            visible_lines = response_lines[:lines_to_show]
            bubble_h = 50 + len(visible_lines) * 28
            
            add_rounded_rect(draw, [80, ai_y, 700, ai_y + bubble_h], 15, fill=WHATSAPP_BUBBLE)
            
            for i, line in enumerate(visible_lines):
                color = ACCENT_GOLD if "Max Healthcare" in line else WHITE
                draw.text((100, ai_y + 15 + i * 28), line, fill=color, font=load_font(16))
        
        # === EVIDENCE BADGE (t=6) - THE KEY FEATURE ===
        if t > 6:
            badge_y = content_y + 360
            badge_alpha = min(1.0, (t - 6) * 2)
            
            # Badge container with glow effect
            glow_size = int(10 * badge_alpha)
            for i in range(glow_size, 0, -2):
                alpha = int(100 * badge_alpha * (i / glow_size))
                glow_color = (52 + alpha//4, 168 + alpha//8, 83)
                add_rounded_rect(draw, [75 - i, badge_y - i, 705 + i, badge_y + 90 + i], 
                               20, fill=None, outline=glow_color, width=2)
            
            # Main badge
            add_rounded_rect(draw, [80, badge_y, 700, badge_y + 80], 18, 
                           fill=(35, 60, 45), outline=ACCENT_GREEN, width=3)
            
            # Badge icon and title
            draw.text((100, badge_y + 12), "✓ HIGH CONFIDENCE", fill=ACCENT_GREEN, font=load_font(18, True))
            
            # Progress bar animation
            bar_y = badge_y + 42
            draw.rectangle([100, bar_y, 400, bar_y + 12], fill=(60, 60, 60), outline=GRAY_DARK)
            confidence = min(0.87, 0.3 + (t - 6) * 0.15) if t > 6 else 0
            bar_width = int(300 * confidence)
            if bar_width > 0:
                draw.rectangle([100, bar_y, 100 + bar_width, bar_y + 12], fill=ACCENT_GREEN)
            
            # Percentage
            draw.text((420, bar_y - 2), f"{int(confidence * 100)}%", fill=ACCENT_GREEN, font=load_font(16, True))
            
            # Source info
            draw.text((100, badge_y + 58), "Source: Max Healthcare Palliative Care DB • 87% match • Last verified: Jan 2026", 
                     fill=GRAY_MID, font=load_font(13))
        
        # === SECONDARY BADGE - WARNING (t=8) ===
        if t > 8:
            warn_y = content_y + 460
            warn_alpha = min(1.0, (t - 8) * 2)
            
            add_rounded_rect(draw, [80, warn_y, 700, warn_y + 60], 15, 
                           fill=(60, 50, 35), outline=ACCENT_ORANGE, width=2)
            draw.text((100, warn_y + 10), "⚠ Physician Consult Recommended", fill=ACCENT_ORANGE, font=load_font(15, True))
            draw.text((100, warn_y + 35), "Breakthrough pain may need dose adjustment", fill=GRAY_LIGHT, font=load_font(14))
        
        # Footer status bar
        draw.rectangle([0, HEIGHT - 30, WIDTH, HEIGHT], fill=(30, 40, 45))
        draw.text((20, HEIGHT - 22), "🔒 End-to-end encrypted • Session ID: ps-2026-0128-x7k9", fill=GRAY_MID, font=font_small)
        draw.text((WIDTH - 200, HEIGHT - 22), "Palli Sahayak v2.1.0", fill=GRAY_MID, font=font_small)
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def scene_medication_reminders(duration=12):
    """Scene 2: Medication Reminder System for Elderly"""
    def make_frame(t):
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_DARK)
        draw = ImageDraw.Draw(img)
        
        # Split screen - Left: Dashboard, Right: Phone call
        
        # === LEFT PANEL - Admin Dashboard ===
        draw.rectangle([0, 0, WIDTH//2 - 10, HEIGHT], fill=(22, 25, 30))
        
        # Dashboard header
        draw.rectangle([0, 0, WIDTH//2 - 10, 60], fill=(35, 45, 55))
        draw.text((30, 18), "📊 Palli Sahayak Admin Dashboard", fill=WHITE, font=load_font(22, True))
        
        # Patient card
        card_y = 80
        add_rounded_rect(draw, [20, card_y, WIDTH//2 - 30, card_y + 180], 15, 
                        fill=(30, 35, 40), outline=(60, 70, 80), width=2)
        
        draw.text((40, card_y + 15), "👤 Patient Profile", fill=ACCENT_BLUE, font=load_font(18, True))
        
        patient_info = [
            ("Name:", "Smt. Lakshmi Devi"),
            ("Age:", "72 years"),
            ("Condition:", "Breast Cancer Stage 2"),
            ("Language:", "Hindi (hi-IN)"),
            ("Caregiver:", "Son - Rajesh Kumar (+91-98xxx-xxxxx)"),
        ]
        
        for i, (label, value) in enumerate(patient_info):
            y = card_y + 50 + i * 25
            draw.text((40, y), label, fill=GRAY_MID, font=load_font(14))
            draw.text((140, y), value, fill=WHITE, font=load_font(14))
        
        # Medication schedule table
        table_y = 280
        draw.text((30, table_y), "💊 Medication Schedule", fill=ACCENT_BLUE, font=load_font(18, True))
        
        # Table header
        header_y = table_y + 40
        draw.rectangle([20, header_y, WIDTH//2 - 30, header_y + 35], fill=(45, 55, 65))
        draw.text((30, header_y + 8), "Time", fill=GRAY_LIGHT, font=load_font(14, True))
        draw.text((120, header_y + 8), "Medication", fill=GRAY_LIGHT, font=load_font(14, True))
        draw.text((280, header_y + 8), "Dose", fill=GRAY_LIGHT, font=load_font(14, True))
        draw.text((380, header_y + 8), "Status", fill=GRAY_LIGHT, font=load_font(14, True))
        
        # Table rows with animation
        meds = [
            ("08:00", "Morphine", "5mg", "completed" if t > 4 else "scheduled"),
            ("14:00", "Ondansetron", "4mg", "upcoming"),
            ("20:00", "Morphine", "5mg", "upcoming"),
        ]
        
        for i, (time, med, dose, status) in enumerate(meds):
            row_y = header_y + 45 + i * 40
            bg_color = (35, 55, 45) if status == "completed" else (40, 45, 50)
            
            # Highlight row if completed
            if status == "completed":
                draw.rectangle([20, row_y, WIDTH//2 - 30, row_y + 35], fill=bg_color)
            
            draw.text((30, row_y + 8), time, fill=WHITE, font=load_font(14))
            draw.text((120, row_y + 8), med, fill=WHITE, font=load_font(14))
            draw.text((280, row_y + 8), dose, fill=GRAY_LIGHT, font=load_font(14))
            
            # Status with color
            status_colors = {
                "completed": (ACCENT_GREEN, "✓ Completed"),
                "scheduled": (ACCENT_ORANGE, "⏳ Scheduled"),
                "upcoming": (GRAY_MID, "⏰ Upcoming")
            }
            color, text = status_colors.get(status, (GRAY_MID, status))
            draw.text((380, row_y + 8), text, fill=color, font=load_font(13))
        
        # Activity log (appears at t=5)
        if t > 5:
            log_y = 500
            draw.text((30, log_y), "📡 Activity Log", fill=ACCENT_BLUE, font=load_font(18, True))
            
            logs = [
                ("08:00:00", "Voice reminder initiated via Bolna.ai"),
                ("08:00:05", "Call connected to +91-98xxx-xxxxx"),
                ("08:00:08", "Played: 'नमस्ते लक्ष्मी जी, दवाई का समय हुआ है'"),
                ("08:00:25", "DTMF '1' received - Patient confirmed"),
                ("08:00:26", "✓ Adherence logged, caregiver notified"),
            ]
            
            for i, (timestamp, log) in enumerate(logs):
                if t > 5.5 + i * 0.5:
                    line_y = log_y + 40 + i * 28
                    draw.text((30, line_y), timestamp, fill=GRAY_MID, font=load_font(12))
                    
                    # Color based on log type
                    log_color = ACCENT_GREEN if "✓" in log else (ACCENT_BLUE if "DTMF" in log else GRAY_LIGHT)
                    draw.text((110, line_y), log, fill=log_color, font=load_font(13))
        
        # === RIGHT PANEL - Phone Call UI ===
        phone_x = WIDTH//2 + 10
        draw.rectangle([phone_x, 0, WIDTH, HEIGHT], fill=(15, 20, 25))
        
        # Phone header
        draw.rectangle([phone_x, 0, WIDTH, 60], fill=(35, 45, 55))
        draw.text((phone_x + 30, 18), "📞 Voice Call Simulation", fill=WHITE, font=load_font(20, True))
        
        # Phone screen
        screen_x = phone_x + 100
        screen_y = 100
        screen_w = 500
        screen_h = 800
        
        # Phone bezel
        add_rounded_rect(draw, [screen_x - 10, screen_y - 10, screen_x + screen_w + 10, screen_y + screen_h + 10], 
                        40, fill=(50, 50, 55), outline=(80, 80, 85), width=3)
        
        # Phone screen
        add_rounded_rect(draw, [screen_x, screen_y, screen_x + screen_w, screen_y + screen_h], 
                        30, fill=(10, 15, 20))
        
        # Caller info
        if t > 1:
            draw.text((screen_x + screen_w//2, screen_y + 80), "पल्ली सहायक", fill=ACCENT_GREEN, 
                     font=load_font(28, True), anchor="mm")
            draw.text((screen_x + screen_w//2, screen_y + 120), "Palli Sahayak", fill=GRAY_MID, 
                     font=load_font(18), anchor="mm")
            draw.text((screen_x + screen_w//2, screen_y + 160), "+91-XXXX-NH-HELP", fill=WHITE, 
                     font=load_font(20), anchor="mm")
        
        # Call status
        if 2 < t < 6:
            pulse = abs(np.sin(t * 4)) 
            status_text = "Calling..." if t < 3 else "Connected"
            color = (100 + int(155 * pulse), 200, 100) if t < 3 else ACCENT_GREEN
            draw.text((screen_x + screen_w//2, screen_y + 220), status_text, fill=color, 
                     font=load_font(16), anchor="mm")
        
        # Audio visualization
        if 3 < t < 7:
            viz_y = screen_y + 300
            bars = 20
            for i in range(bars):
                bar_h = 20 + int(60 * abs(np.sin(t * 8 + i * 0.5)) * (0.5 + 0.5 * np.sin(i)))
                bar_x = screen_x + 50 + i * 20
                color_val = 100 + int(155 * (bar_h / 80))
                draw.rectangle([bar_x, viz_y + 80 - bar_h, bar_x + 12, viz_y + 80], 
                             fill=(0, color_val//2, color_val//3))
        
        # Voice message transcription
        if t > 3.5:
            msg_y = screen_y + 420
            messages = [
                (3.5, "नमस्ते लक्ष्मी जी..."),
                (4.5, "मैं पल्ली सहायक बोल रहा हूं"),
                (5.0, "आपकी दवाई का समय हुआ है"),
                (5.5, "कृपया मोर्फीन 5mg लें"),
                (6.0, "लेने के बाद 1 दबाएं"),
            ]
            
            for timestamp, msg in messages:
                if t > timestamp:
                    add_rounded_rect(draw, [screen_x + 30, msg_y, screen_x + screen_w - 30, msg_y + 40], 
                                   15, fill=(30, 50, 45))
                    draw.text((screen_x + 45, msg_y + 10), msg, fill=WHITE, font=load_font(15))
                    msg_y += 50
        
        # DTMF keypad
        if t > 6:
            keypad_y = screen_y + 650
            # Highlight button 1
            btn_x = screen_x + screen_w//2 - 35
            pulse = 1.0 + 0.1 * np.sin(t * 10) if t > 7 else 1.0
            btn_size = int(70 * pulse)
            
            add_rounded_rect(draw, [btn_x, keypad_y, btn_x + btn_size, keypad_y + btn_size], 
                           35, fill=ACCENT_GREEN, outline=WHITE, width=3)
            draw.text((btn_x + btn_size//2, keypad_y + btn_size//2 - 5), "1", fill=WHITE, 
                     font=load_font(32, True), anchor="mm")
            draw.text((btn_x + btn_size//2, keypad_y + btn_size//2 + 20), " Taken", fill=(200, 255, 200), 
                     font=load_font(11), anchor="mm")
        
        # Call ended status
        if t > 8:
            draw.text((screen_x + screen_w//2, screen_y + 750), "✓ Call Completed • 28 seconds", 
                     fill=ACCENT_GREEN, font=load_font(14), anchor="mm")
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def scene_warm_handoff(duration=15):
    """Scene 3: SIP-REFER Warm Handoff to Human Agent"""
    def make_frame(t):
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_DARK)
        draw = ImageDraw.Draw(img)
        
        # Header
        draw.rectangle([0, 0, WIDTH, 70], fill=(45, 35, 55))
        draw.text((30, 20), "🤝 Synergistic Human-AI Collaboration", fill=ACCENT_GOLD, font=load_font(26, True))
        draw.text((WIDTH - 350, 25), "SIP-REFER via Vobiz.ai • LiveKit", fill=GRAY_MID, font=load_font(16))
        
        # Main content area - split into 3 columns
        col_width = WIDTH // 3 - 20
        
        # === COLUMN 1: AI Chat (Left) ===
        col1_x = 20
        draw.rectangle([col1_x, 90, col1_x + col_width, HEIGHT - 50], fill=(25, 30, 35))
        draw.text((col1_x + 15, 105), "🤖 AI Conversation", fill=ACCENT_BLUE, font=load_font(16, True))
        
        # Crisis trigger message
        if t > 1:
            chat_y = 150
            # User message
            add_rounded_rect(draw, [col1_x + 10, chat_y, col1_x + col_width - 10, chat_y + 80], 
                           12, fill=(60, 40, 40), outline=ACCENT_RED, width=2)
            draw.text((col1_x + 20, chat_y + 10), "Patient:", fill=ACCENT_RED, font=load_font(13, True))
            
            msg_text = "I can't bear this pain anymore..."
            if t > 1.5:
                msg_text += "\nI want to give up. What's the point?"
            draw.text((col1_x + 20, chat_y + 30), msg_text, fill=WHITE, font=load_font(14))
        
        # AI Analysis
        if t > 3:
            analysis_y = 250
            add_rounded_rect(draw, [col1_x + 10, analysis_y, col1_x + col_width - 10, analysis_y + 120], 
                           12, fill=(40, 35, 30), outline=ACCENT_ORANGE, width=2)
            draw.text((col1_x + 20, analysis_y + 10), "⚠️ AI Analysis:", fill=ACCENT_ORANGE, font=load_font(14, True))
            
            indicators = [
                ("Depression keywords detected", t > 3.5),
                ("Suicidal ideation risk: MEDIUM", t > 4),
                ("Confidence: 38% (Below threshold)", t > 4.5),
                ("→ Escalation recommended", t > 5),
            ]
            
            for i, (text, show) in enumerate(indicators):
                if show:
                    y = analysis_y + 40 + i * 22
                    draw.text((col1_x + 25, y), "• " + text, fill=GRAY_LIGHT, font=load_font(13))
        
        # === COLUMN 2: SIP-REFER Flow (Center) ===
        col2_x = col_width + 40
        draw.rectangle([col2_x, 90, col2_x + col_width, HEIGHT - 50], fill=(30, 30, 35))
        draw.text((col2_x + 15, 105), "🔄 SIP Call Transfer", fill=ACCENT_GOLD, font=load_font(16, True))
        
        # SIP Flow diagram
        flow_y = 150
        steps = [
            ("1. TRIGGER", "Crisis detected\nAuto-escalation", 2, ACCENT_RED),
            ("2. SIP-REFER", "REFER sip:counselor@\npalliative.care SIP/2.0", 4, ACCENT_BLUE),
            ("3. QUEUE", "Finding available\npsychologist...", 6, ACCENT_ORANGE),
            ("4. CONNECT", "Dr. Priya Sharma\nPsychologist - Available", 9, ACCENT_GREEN),
        ]
        
        for title, desc, trigger_time, color in steps:
            if t > trigger_time:
                step_h = 90
                # Step box
                is_active = t < trigger_time + 2
                bg_color = (color[0]//4, color[1]//4, color[2]//4) if is_active else (35, 40, 45)
                border_width = 3 if is_active else 1
                
                add_rounded_rect(draw, [col2_x + 10, flow_y, col2_x + col_width - 10, flow_y + step_h], 
                               12, fill=bg_color, outline=color, width=border_width)
                
                draw.text((col2_x + 20, flow_y + 10), title, fill=color, font=load_font(14, True))
                draw.text((col2_x + 20, flow_y + 35), desc, fill=WHITE, font=load_font(13))
                
                # Connection arrow
                if title != "4. CONNECT":
                    arrow_y = flow_y + step_h + 5
                    draw.polygon([
                        (col2_x + col_width//2, arrow_y + 15),
                        (col2_x + col_width//2 - 10, arrow_y),
                        (col2_x + col_width//2 + 10, arrow_y)
                    ], fill=color)
                
                flow_y += step_h + 25
        
        # Context transfer animation
        if t > 10:
            ctx_y = 580
            add_rounded_rect(draw, [col2_x + 10, ctx_y, col2_x + col_width - 10, ctx_y + 150], 
                           12, fill=(35, 50, 40), outline=ACCENT_GREEN, width=2)
            draw.text((col2_x + 20, ctx_y + 10), "📋 Context Transfer", fill=ACCENT_GREEN, font=load_font(14, True))
            
            ctx_items = [
                ("Patient: Rajesh Kumar, 65", 10.5),
                ("Cancer: Lung, Stage 3", 11),
                ("Issue: Breakthrough pain + Depression", 11.5),
                ("AI chat: 8min, 12 exchanges", 12),
                ("Risk: Medium (suicidal keywords)", 12.5),
            ]
            
            for text, trigger in ctx_items:
                if t > trigger:
                    idx = int((trigger - 10) * 2)
                    draw.text((col2_x + 25, ctx_y + 40 + idx * 22), "✓ " + text, 
                             fill=GRAY_LIGHT, font=load_font(12))
        
        # === COLUMN 3: Human Agent (Right) ===
        col3_x = col_width * 2 + 60
        draw.rectangle([col3_x, 90, col3_x + col_width, HEIGHT - 50], fill=(25, 35, 30))
        draw.text((col3_x + 15, 105), "👩‍⚕️ Human Agent", fill=ACCENT_GREEN, font=load_font(16, True))
        
        # Agent profile
        if t > 9:
            profile_y = 150
            add_rounded_rect(draw, [col3_x + 10, profile_y, col3_x + col_width - 10, profile_y + 120], 
                           12, fill=(35, 50, 45), outline=ACCENT_GREEN, width=2)
            
            # Avatar circle
            draw.ellipse([col3_x + 20, profile_y + 15, col3_x + 70, profile_y + 65], fill=(100, 150, 120))
            draw.text((col3_x + 35, profile_y + 25), "👩", fill=WHITE, font=load_font(30))
            
            draw.text((col3_x + 85, profile_y + 20), "Dr. Priya Sharma", fill=WHITE, font=load_font(16, True))
            draw.text((col3_x + 85, profile_y + 45), "Palliative Psychologist", fill=GRAY_MID, font=load_font(13))
            draw.text((col3_x + 85, profile_y + 68), "🟢 Online • 5+ years exp", fill=ACCENT_GREEN, font=load_font(12))
        
        # Agent chat
        if t > 11:
            chat_y = 290
            add_rounded_rect(draw, [col3_x + 10, chat_y, col3_x + col_width - 10, chat_y + 80], 
                           12, fill=(45, 75, 60))
            draw.text((col3_x + 20, chat_y + 10), "Namaste Rajesh ji...", fill=WHITE, font=load_font(14))
            draw.text((col3_x + 20, chat_y + 40), "I'm Dr. Priya. I understand", fill=GRAY_LIGHT, font=load_font(13))
            draw.text((col3_x + 20, chat_y + 60), "you're going through a lot.", fill=GRAY_LIGHT, font=load_font(13))
        
        # Warm handoff complete
        if t > 13:
            complete_y = 400
            add_rounded_rect(draw, [col3_x + 10, complete_y, col3_x + col_width - 10, complete_y + 100], 
                           15, fill=(45, 80, 50), outline=ACCENT_GREEN, width=3)
            draw.text((col3_x + col_width//2, complete_y + 25), "✓ WARM HANDOFF COMPLETE", 
                     fill=ACCENT_GREEN, font=load_font(15, True), anchor="mm")
            draw.text((col3_x + col_width//2, complete_y + 55), "AI → Human • Context preserved", 
                     fill=GRAY_LIGHT, font=load_font(12), anchor="mm")
            draw.text((col3_x + col_width//2, complete_y + 80), "Session transferred 10:42 AM", 
                     fill=GRAY_MID, font=load_font(11), anchor="mm")
        
        # Bottom synergy banner
        if t > 7:
            banner_y = HEIGHT - 120
            alpha = min(1.0, (t - 7) * 0.5)
            banner_color = (int(60 * alpha), int(50 * alpha), int(40 * alpha))
            
            draw.rectangle([0, banner_y, WIDTH, HEIGHT - 50], fill=banner_color)
            
            benefits = [
                "AI: 24/7 routine support • Medication • Pain guidance",
                "Human: Emotional care • Complex cases • Ethics",
                "SIP-REFER: Seamless warm handoff • Zero data loss"
            ]
            
            x_pos = 50
            for benefit in benefits:
                draw.text((x_pos, banner_y + 25), "✦ " + benefit, fill=ACCENT_GOLD, font=load_font(13))
                x_pos += 500
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def scene_closing(duration=6):
    """Final scene with summary"""
    def make_frame(t):
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_DARK)
        draw = ImageDraw.Draw(img)
        
        # Title
        draw.text((WIDTH//2, 120), "Palli Sahayak", fill=ACCENT_GOLD, font=load_font(56, True), anchor="mm")
        draw.text((WIDTH//2, 200), "Production-Ready Safety Features", fill=WHITE, font=load_font(28), anchor="mm")
        draw.text((WIDTH//2, 250), "Deployed January 28, 2026", fill=GRAY_MID, font=load_font(18), anchor="mm")
        
        # Feature cards
        features = [
            ("🔬", "Evidence Badges", "87% confidence • Source verified", ACCENT_GREEN),
            ("💊", "Voice Reminders", "DTMF confirmation • Adherence tracked", ACCENT_BLUE),
            ("🤝", "SIP-REFER Handoff", "Vobiz.ai • Warm transfer • Context preserved", ACCENT_GOLD),
            ("🚨", "Emergency Detection", "5 languages • Auto-escalation", ACCENT_RED),
            ("📏", "Smart Responses", "Adaptive • Voice-optimized", ACCENT_ORANGE),
        ]
        
        card_width = 350
        start_x = (WIDTH - (5 * card_width + 4 * 30)) // 2
        
        for i, (emoji, title, desc, color) in enumerate(features):
            x = start_x + i * (card_width + 30)
            
            # Stagger animation
            if t > i * 0.3:
                alpha = min(1.0, (t - i * 0.3) * 2)
                
                # Card background
                card_color = (int(40 * alpha), int(40 * alpha), int(45 * alpha))
                add_rounded_rect(draw, [x, 320, x + card_width, 550], 20, 
                               fill=card_color, outline=(int(c[0] * alpha) for c in color), width=3)
                
                # Emoji
                draw.text((x + card_width//2, 380), emoji, fill=(int(c[0] * alpha) for c in color), 
                         font=load_font(48), anchor="mm")
                
                # Title
                draw.text((x + card_width//2, 450), title, fill=(int(255 * alpha),) * 3, 
                         font=load_font(18, True), anchor="mm")
                
                # Description
                draw.text((x + card_width//2, 500), desc, fill=(int(200 * alpha),) * 3, 
                         font=load_font(13), anchor="mm")
        
        # Bottom message
        if t > 3:
            draw.text((WIDTH//2, 700), "Human-AI Synergy for Better Palliative Care", 
                     fill=ACCENT_GOLD, font=load_font(24), anchor="mm")
            draw.text((WIDTH//2, 850), "inventcures.github.io/palli-sahayak", 
                     fill=GRAY_MID, font=load_font(18), anchor="mm")
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def main():
    """Generate the system demo video"""
    print("=" * 70)
    print("🎬 Palli Sahayak - System Demo Video Generator")
    print("=" * 70)
    print()
    
    scenes = [
        ("Evidence Badges (WhatsApp UI)", scene_evidence_badges(12)),
        ("Medication Reminders (Dashboard)", scene_medication_reminders(12)),
        ("SIP-REFER Warm Handoff", scene_warm_handoff(15)),
        ("Summary", scene_closing(6)),
    ]
    
    video_clips = []
    for name, clip in scenes:
        print(f"  Rendering: {name}...")
        clip = FadeIn(0.5).apply(clip)
        clip = FadeOut(0.5).apply(clip)
        video_clips.append(clip)
    
    print("\n🔄 Concatenating scenes...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    output_path = "palli_sahayak_system_demo.mp4"
    print(f"\n💾 Saving to: {output_path}")
    print("   Encoding video (this may take a minute)...")
    
    final_video.write_videofile(
        output_path,
        fps=FPS,
        codec='libx264',
        audio=False,
        preset='medium',
        threads=4
    )
    
    # Cleanup
    final_video.close()
    for clip in video_clips:
        clip.close()
    
    print("\n" + "=" * 70)
    print(f"✅ System demo video generated!")
    print(f"📁 Output: {output_path}")
    print(f"⏱️  Duration: 45 seconds")
    print(f"📺 Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps")
    print()
    print("🎥 Scenes:")
    print("   1. WhatsApp UI with Evidence Badges (87% confidence)")
    print("   2. Admin Dashboard + Voice Call for Medication Reminders")
    print("   3. SIP-REFER Warm Handoff (Vobiz.ai integration)")
    print("   4. Feature Summary")
    print("=" * 70)

if __name__ == "__main__":
    main()
