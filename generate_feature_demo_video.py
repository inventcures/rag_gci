#!/usr/bin/env python3
"""
Palli Sahayak - Feature Demo Video Generator
============================================

Generates an MP4 video showcasing:
1. Evidence Badges (confidence scores, source quality)
2. Medication Voice Reminders for elderly patients
3. Warm Handoff to human agents using SIP-REFER (Vobiz.ai)

Uses MoviePy for video generation with simulated UI screens.
"""

import os
import sys
from moviepy import VideoClip, ColorClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Video settings
WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION_PER_SCENE = 8  # seconds

# Colors
DARK_BG = (45, 35, 30)
PRIMARY_BROWN = (107, 78, 61)
ACCENT_GOLD = (245, 193, 108)
WHITE = (255, 255, 255)
GREEN = (76, 175, 80)
RED = (244, 67, 54)
BLUE = (33, 150, 243)
ORANGE = (255, 152, 0)

def create_gradient_background(width, height, color1, color2):
    """Create a gradient background"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return np.array(img)

def create_title_card(title, subtitle, duration=4):
    """Create title card scene"""
    def make_frame(t):
        img = Image.new('RGB', (WIDTH, HEIGHT), DARK_BG)
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fallback to default
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
            subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Logo/title area
        draw.text((WIDTH//2, 300), "पल्ली सहायक", fill=ACCENT_GOLD, font=title_font, anchor="mm")
        draw.text((WIDTH//2, 420), "Palli Sahayak", fill=WHITE, font=title_font, anchor="mm")
        
        # Subtitle
        draw.text((WIDTH//2, 550), title, fill=ACCENT_GOLD, font=subtitle_font, anchor="mm")
        draw.text((WIDTH//2, 620), subtitle, fill=(180, 180, 180), font=small_font, anchor="mm")
        
        # Date
        draw.text((WIDTH//2, 900), "EkStep Voice AI Event • January 28, 2026", fill=(120, 120, 120), font=small_font, anchor="mm")
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def create_evidence_badges_scene(duration=8):
    """Scene 1: Evidence Badges in Action"""
    def make_frame(t):
        img = Image.new('RGB', (WIDTH, HEIGHT), DARK_BG)
        draw = ImageDraw.Draw(img)
        
        try:
            header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            badge_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            header_font = text_font = small_font = badge_font = ImageFont.load_default()
        
        # Header
        draw.text((60, 50), "🔬 FEATURE 1: Evidence-Based Confidence Badges", fill=ACCENT_GOLD, font=header_font)
        draw.text((60, 110), "Every response shows source quality and confidence score", fill=(180, 180, 180), font=small_font)
        
        # Chat UI simulation
        chat_bg = (35, 30, 28)
        draw.rounded_rectangle([100, 180, 900, 650], radius=20, fill=chat_bg, outline=PRIMARY_BROWN, width=2)
        
        # User query
        user_bubble = (60, 100, 200)
        draw.rounded_rectangle([140, 220, 600, 290], radius=15, fill=user_bubble)
        draw.text((160, 235), "Patient has severe pain after chemotherapy.", fill=WHITE, font=text_font)
        draw.text((160, 265), "What medication should I give?", fill=WHITE, font=text_font)
        
        # AI Response with animation
        if t > 1:
            ai_bubble = (45, 55, 60)
            draw.rounded_rectangle([140, 330, 800, 520], radius=15, fill=ai_bubble)
            
            # Response text with typewriter effect
            response = "Based on similar cases from Max Healthcare..."
            if t > 1.5:
                draw.text((160, 350), response, fill=WHITE, font=text_font)
            if t > 2:
                draw.text((160, 390), "• Morphine 5-10mg oral every 4h PRN", fill=WHITE, font=text_font)
            if t > 2.5:
                draw.text((160, 430), "• Monitor for constipation, sedation", fill=WHITE, font=text_font)
            if t > 3:
                draw.text((160, 470), "• Consult physician if pain persists >24h", fill=WHITE, font=text_font)
        
        # Evidence Badge Panel (animates in at t=3.5)
        if t > 3.5:
            badge_alpha = min(255, int((t - 3.5) * 255))
            badge_y = 550
            
            # Badge background
            badge_bg = (50, 70, 50)
            draw.rounded_rectangle([140, badge_y, 800, badge_y + 80], radius=10, fill=badge_bg, outline=GREEN, width=2)
            
            # Badge content
            draw.text((160, badge_y + 15), "✓ HIGH CONFIDENCE", fill=GREEN, font=badge_font)
            draw.text((160, badge_y + 45), "Source: Max Healthcare Case Vignettes (87% match)", fill=(200, 200, 200), font=small_font)
            
            # Confidence meter
            meter_width = 200
            draw.rectangle([550, badge_y + 20, 550 + meter_width, badge_y + 40], fill=(60, 60, 60), outline=(100, 100, 100))
            confidence = min(0.87, (t - 3.5) * 0.2)
            draw.rectangle([550, badge_y + 20, 550 + int(meter_width * confidence), badge_y + 40], fill=GREEN)
            draw.text((550 + meter_width + 10, badge_y + 18), "87%", fill=GREEN, font=badge_font)
        
        # Side panel with badge types
        if t > 5:
            draw.rounded_rectangle([950, 180, 1820, 650], radius=20, fill=chat_bg, outline=PRIMARY_BROWN, width=2)
            draw.text((980, 210), "Evidence Badge Types:", fill=ACCENT_GOLD, font=text_font)
            
            badge_types = [
                ("🟢 HIGH", "85-100% | Trusted sources", GREEN),
                ("🟡 MEDIUM", "60-84% | Review needed", ORANGE),
                ("🔴 LOW", "<60% | Physician consult", RED),
            ]
            
            for i, (label, desc, color) in enumerate(badge_types):
                y_pos = 280 + i * 100
                draw.rounded_rectangle([980, y_pos, 1800, y_pos + 70], radius=10, fill=(40, 40, 40), outline=color, width=1)
                draw.text((1000, y_pos + 10), label, fill=color, font=text_font)
                draw.text((1000, y_pos + 40), desc, fill=(180, 180, 180), font=small_font)
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def create_medication_reminders_scene(duration=8):
    """Scene 2: Medication Voice Reminders for Elderly"""
    def make_frame(t):
        img = Image.new('RGB', (WIDTH, HEIGHT), DARK_BG)
        draw = ImageDraw.Draw(img)
        
        try:
            header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            phone_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            header_font = text_font = small_font = phone_font = ImageFont.load_default()
        
        # Header
        draw.text((60, 50), "💊 FEATURE 2: Voice Medication Reminders", fill=ACCENT_GOLD, font=header_font)
        draw.text((60, 110), "Automated voice calls for elderly patients • DTMF/Voice confirmation", fill=(180, 180, 180), font=small_font)
        
        # Phone UI simulation
        phone_bg = (30, 30, 35)
        draw.rounded_rectangle([500, 200, 900, 900], radius=40, fill=phone_bg, outline=(80, 80, 80), width=3)
        
        # Phone status bar
        draw.rectangle([500, 200, 900, 240], fill=(20, 20, 25))
        draw.text((700, 220), "9:00 AM", fill=WHITE, font=small_font, anchor="mm")
        
        # Caller ID
        if t > 0.5:
            draw.text((700, 320), "पल्ली सहायक", fill=ACCENT_GOLD, font=text_font, anchor="mm")
            draw.text((700, 360), "Palli Sahayak", fill=(180, 180, 180), font=small_font, anchor="mm")
            draw.text((700, 420), "+91-XXXX-NH-HELP", fill=BLUE, font=text_font, anchor="mm")
        
        # Animated call button
        if 1 < t < 6:
            pulse = abs(np.sin(t * 3)) * 20
            accept_color = (76, 175, 80)
            draw.ellipse([620 - pulse, 500 - pulse, 780 + pulse, 660 + pulse], outline=accept_color, width=3)
            draw.ellipse([620, 500, 780, 660], fill=accept_color)
            draw.text((700, 580), "📞", fill=WHITE, font=header_font, anchor="mm")
        
        # Call connected
        if t >= 6:
            draw.text((700, 580), "✓ Connected", fill=GREEN, font=text_font, anchor="mm")
        
        # Side panels showing reminder system
        # Patient profile
        if t > 1:
            draw.rounded_rectangle([50, 200, 450, 500], radius=15, fill=(35, 35, 40), outline=PRIMARY_BROWN, width=2)
            draw.text((70, 230), "👤 Patient Profile", fill=ACCENT_GOLD, font=text_font)
            draw.text((70, 290), "Name: Smt. Lakshmi Devi", fill=WHITE, font=small_font)
            draw.text((70, 330), "Age: 72 years", fill=WHITE, font=small_font)
            draw.text((70, 370), "Condition: Cancer - Stage 2", fill=WHITE, font=small_font)
            draw.text((70, 410), "Language: Hindi", fill=WHITE, font=small_font)
            draw.text((70, 450), "Caregiver: Son (primary)", fill=WHITE, font=small_font)
        
        # Medication schedule
        if t > 2:
            draw.rounded_rectangle([50, 520, 450, 900], radius=15, fill=(35, 35, 40), outline=PRIMARY_BROWN, width=2)
            draw.text((70, 550), "📋 Today's Medications", fill=ACCENT_GOLD, font=text_font)
            
            meds = [
                ("8:00 AM", "Morphine 5mg", "⏳ Pending" if t < 6 else "✓ Taken"),
                ("2:00 PM", "Ondansetron 4mg", "⏰ Upcoming"),
                ("8:00 PM", "Morphine 5mg", "⏰ Upcoming"),
            ]
            
            for i, (time, med, status) in enumerate(meds):
                y = 610 + i * 80
                status_color = GREEN if "✓" in status else (ORANGE if "⏰" in status else (200, 200, 200))
                draw.text((70, y), f"{time}: {med}", fill=WHITE, font=small_font)
                draw.text((70, y + 30), status, fill=status_color, font=small_font)
        
        # System logs
        if t > 3:
            draw.rounded_rectangle([950, 200, 1870, 600], radius=15, fill=(25, 30, 35), outline=(50, 60, 70), width=2)
            draw.text((970, 230), "📡 System Activity Log", fill=ACCENT_GOLD, font=text_font)
            
            logs = [
                ("08:00:00", "Reminder scheduled", (200, 200, 200)),
                ("08:00:05", "Initiating Bolna.ai call", BLUE),
                ("08:00:08", "Patient answered", GREEN),
                ("08:00:15", "Voice: नमस्ते, दवाई का समय हुआ है", (200, 200, 200)),
            ]
            
            for i, (timestamp, log, color) in enumerate(logs):
                if t > 3 + i * 0.8:
                    y = 290 + i * 50
                    draw.text((970, y), timestamp, fill=(100, 100, 100), font=small_font)
                    draw.text((1100, y), log, fill=color, font=small_font)
        
        # Confirmation UI
        if t > 5:
            draw.rounded_rectangle([950, 620, 1870, 900], radius=15, fill=(35, 50, 35), outline=GREEN, width=2)
            draw.text((970, 650), "✓ DTMF Confirmation Received", fill=GREEN, font=text_font)
            draw.text((970, 710), "Patient pressed 1 - Medication taken", fill=WHITE, font=small_font)
            draw.text((970, 760), "Caregiver notified via WhatsApp", fill=WHITE, font=small_font)
            draw.text((970, 810), "Adherence logged to dashboard", fill=WHITE, font=small_font)
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def create_handoff_scene(duration=10):
    """Scene 3: Warm Handoff with SIP-REFER (Vobiz.ai)"""
    def make_frame(t):
        img = Image.new('RGB', (WIDTH, HEIGHT), DARK_BG)
        draw = ImageDraw.Draw(img)
        
        try:
            header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            code_font = ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", 16)
        except:
            header_font = text_font = small_font = code_font = ImageFont.load_default()
        
        # Header with emphasis
        draw.text((60, 40), "🤝 FEATURE 3: Synergistic Human-AI Collaboration", fill=ACCENT_GOLD, font=header_font)
        
        # Subheader with SIP-REFER mention
        if t > 0.5:
            draw.text((60, 100), "Warm Handoff via SIP-REFER • Vobiz.ai • LiveKit Integration", fill=BLUE, font=text_font)
        
        # Scenario box
        if t > 1:
            draw.rounded_rectangle([60, 160, 900, 350], radius=15, fill=(45, 35, 35), outline=RED, width=2)
            draw.text((80, 180), "⚠️ TRIGGER: Emotional Crisis Detected", fill=RED, font=text_font)
            draw.text((80, 240), "Patient: \"I can't bear this pain anymore... I want to give up\"", fill=WHITE, font=text_font)
            draw.text((80, 280), "AI Confidence: LOW (38%) - Mental health support needed", fill=ORANGE, font=small_font)
            draw.text((80, 320), "Auto-triggering warm handoff to counselor...", fill=ACCENT_GOLD, font=small_font)
        
        # SIP-REFER Flow Diagram
        if t > 3:
            # Title
            draw.text((1000, 160), "SIP-REFER Call Transfer Flow", fill=ACCENT_GOLD, font=text_font)
            
            # Boxes for flow
            box_y = 220
            box_height = 70
            
            # Step 1: AI detects escalation
            if t > 3:
                alpha = min(255, int((t - 3) * 255))
                draw.rounded_rectangle([1000, box_y, 1500, box_y + box_height], radius=10, 
                                      fill=(40, 40, 50), outline=BLUE, width=2)
                draw.text((1020, box_y + 20), "1. AI Detects Escalation", fill=WHITE, font=small_font)
                draw.text((1020, box_y + 45), "Depression keywords + Low confidence", fill=(180, 180, 180), font=code_font)
            
            # Arrow
            if t > 3.5:
                draw.polygon([(1250, box_y + box_height), (1240, box_y + box_height + 20), (1260, box_y + box_height + 20)], fill=ACCENT_GOLD)
            
            # Step 2: SIP-REFER initiated
            box_y += 100
            if t > 3.5:
                draw.rounded_rectangle([1000, box_y, 1500, box_y + box_height], radius=10,
                                      fill=(40, 50, 40), outline=GREEN, width=2)
                draw.text((1020, box_y + 15), "2. SIP-REFER to Vobiz.ai", fill=WHITE, font=small_font)
                # Code snippet
                sip_code = 'REFER sip:counselor@palliative.care SIP/2.0'
                draw.text((1020, box_y + 40), sip_code, fill=ACCENT_GOLD, font=code_font)
            
            # Arrow
            if t > 4:
                draw.polygon([(1250, box_y + box_height), (1240, box_y + box_height + 20), (1260, box_y + box_height + 20)], fill=ACCENT_GOLD)
            
            # Step 3: Human agent availability check
            box_y += 100
            if t > 4:
                draw.rounded_rectangle([1000, box_y, 1500, box_y + box_height], radius=10,
                                      fill=(50, 40, 40), outline=ORANGE, width=2)
                draw.text((1020, box_y + 15), "3. Queue & Agent Selection", fill=WHITE, font=small_font)
                draw.text((1020, box_y + 40), "Dr. Priya Sharma (Psychologist) - Available", fill=GREEN, font=code_font)
            
            # Arrow
            if t > 4.5:
                draw.polygon([(1250, box_y + box_height), (1240, box_y + box_height + 20), (1260, box_y + box_height + 20)], fill=ACCENT_GOLD)
            
            # Step 4: Warm handoff
            box_y += 100
            if t > 4.5:
                draw.rounded_rectangle([1000, box_y, 1500, box_y + box_height], radius=10,
                                      fill=(60, 45, 45), outline=ACCENT_GOLD, width=3)
                draw.text((1020, box_y + 15), "4. WARM HANDOFF ✓", fill=ACCENT_GOLD, font=text_font)
                draw.text((1020, box_y + 45), "Context transferred • Patient history • AI summary", fill=WHITE, font=code_font)
        
        # Context transfer panel
        if t > 6:
            draw.rounded_rectangle([60, 380, 900, 700], radius=15, fill=(30, 35, 40), outline=BLUE, width=2)
            draw.text((80, 400), "📋 Context Transferred to Human Agent", fill=BLUE, font=text_font)
            
            context_items = [
                "Patient: Rajesh Kumar, 65, Lung Cancer Stage 3",
                "Current Issue: Breakthrough pain + Depression indicators",
                "AI Interaction: 8 minutes, 12 exchanges",
                "Recommendations given: Morphine protocol (confidence: 38%)",
                "⚠️ Escalation reason: Suicidal ideation keywords detected",
                "Suggested action: Palliative psychology consult + Pain review",
            ]
            
            for i, item in enumerate(context_items):
                y = 460 + i * 35
                color = RED if "⚠️" in item else (WHITE if i < 4 else ACCENT_GOLD)
                draw.text((80, y), item, fill=color, font=small_font)
        
        # Human agent UI
        if t > 7:
            draw.rounded_rectangle([60, 720, 900, 1050], radius=15, fill=(35, 45, 35), outline=GREEN, width=2)
            draw.text((80, 740), "👩‍⚕️ Dr. Priya Sharma (Psychologist) - Connected", fill=GREEN, font=text_font)
            
            # Chat simulation
            if t > 7.5:
                draw.rounded_rectangle([80, 800, 500, 850], radius=10, fill=(60, 100, 60))
                draw.text((100, 815), "Namaste Rajesh ji, I'm Dr. Priya...", fill=WHITE, font=small_font)
            
            if t > 8:
                draw.text((80, 880), "✓ AI Summary reviewed", fill=(180, 180, 180), font=small_font)
                draw.text((80, 910), "✓ Patient history accessed", fill=(180, 180, 180), font=small_font)
                draw.text((80, 940), "✓ Warm handoff complete - Taking over", fill=GREEN, font=small_font)
            
            if t > 8.5:
                draw.rounded_rectangle([400, 970, 880, 1040], radius=10, fill=(80, 60, 40))
                draw.text((420, 985), "AI Handoff Complete", fill=ACCENT_GOLD, font=small_font)
                draw.text((420, 1010), "Session transferred to human agent", fill=(180, 180, 180), font=code_font)
        
        # Synergy highlight
        if t > 9:
            synergy_box = (950, 720, 1870, 1050)
            draw.rounded_rectangle(synergy_box, radius=20, fill=(60, 50, 40), outline=ACCENT_GOLD, width=4)
            
            draw.text((1100, 760), "🎯 SYNERGISTIC COLLABORATION", fill=ACCENT_GOLD, font=header_font)
            
            benefits = [
                "• AI handles 80% routine queries (pain, meds, side effects)",
                "• Human experts handle complex emotional/psychological cases",
                "• Seamless SIP-REFER transfer - no data loss",
                "• Full context preservation for continuity of care",
                "• 24/7 AI + Business hours human specialists",
            ]
            
            for i, benefit in enumerate(benefits):
                y = 830 + i * 40
                draw.text((980, y), benefit, fill=WHITE, font=small_font)
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def create_closing_scene(duration=5):
    """Final scene with all features summary"""
    def make_frame(t):
        img = Image.new('RGB', (WIDTH, HEIGHT), DARK_BG)
        draw = ImageDraw.Draw(img)
        
        try:
            header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 56)
            text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            header_font = text_font = small_font = ImageFont.load_default()
        
        # Title
        draw.text((WIDTH//2, 120), "Palli Sahayak - Production Ready", fill=ACCENT_GOLD, font=header_font, anchor="mm")
        draw.text((WIDTH//2, 190), "5 Safety Features Deployed Today", fill=WHITE, font=text_font, anchor="mm")
        
        # Feature boxes
        features = [
            ("🔬", "Evidence Badges", "Confidence scores & source quality", GREEN),
            ("💊", "Voice Reminders", "Automated calls with DTMF confirm", BLUE),
            ("🤝", "Warm Handoff", "SIP-REFER to human agents", ACCENT_GOLD),
            ("🚨", "Emergency Detection", "5 languages, auto-escalation", RED),
            ("📏", "Smart Responses", "Adaptive length & complexity", ORANGE),
        ]
        
        box_width = 350
        start_x = (WIDTH - (5 * box_width + 4 * 30)) // 2
        
        for i, (emoji, title, desc, color) in enumerate(features):
            x = start_x + i * (box_width + 30)
            
            # Animation stagger
            if t > i * 0.3:
                alpha = min(1.0, (t - i * 0.3) * 2)
                
                # Box
                draw.rounded_rectangle([x, 300, x + box_width, 700], radius=20, 
                                      fill=(40, 40, 45), outline=color, width=3)
                
                # Emoji
                draw.text((x + box_width//2, 380), emoji, fill=color, font=header_font, anchor="mm")
                
                # Title
                draw.text((x + box_width//2, 480), title, fill=WHITE, font=text_font, anchor="mm")
                
                # Description
                draw.text((x + box_width//2, 550), desc, fill=(180, 180, 180), font=small_font, anchor="mm")
        
        # Bottom tagline
        if t > 3:
            draw.text((WIDTH//2, 850), "Human-AI Synergy for Better Palliative Care", fill=ACCENT_GOLD, font=text_font, anchor="mm")
            draw.text((WIDTH//2, 920), "https://inventcures.github.io/palli-sahayak", fill=(150, 150, 150), font=small_font, anchor="mm")
        
        return np.array(img)
    
    return VideoClip(make_frame, duration=duration)

def main():
    """Generate the demo video"""
    print("🎬 Generating Palli Sahayak Feature Demo Video...")
    print("=" * 60)
    
    # Create scenes
    scenes = [
        ("Title", create_title_card("New Features Demo", "Safety & Human-AI Collaboration", 4)),
        ("Evidence Badges", create_evidence_badges_scene(8)),
        ("Medication Reminders", create_medication_reminders_scene(8)),
        ("Warm Handoff", create_handoff_scene(10)),
        ("Closing", create_closing_scene(5)),
    ]
    
    # Process each scene
    video_clips = []
    for name, clip in scenes:
        print(f"  ✓ Rendering: {name}")
        # Apply fade effects
        clip = FadeIn(0.5).apply(clip)
        clip = FadeOut(0.5).apply(clip)
        video_clips.append(clip)
    
    # Concatenate all scenes
    print("\n🔄 Concatenating scenes...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    # Save video
    output_path = "palli_sahayak_features_demo.mp4"
    print(f"\n💾 Saving to: {output_path}")
    print("   This may take a minute...")
    
    final_video.write_videofile(
        output_path,
        fps=FPS,
        codec='libx264',
        audio=False,
        preset='medium',
        threads=4
    )
    
    # Clean up
    final_video.close()
    for clip in video_clips:
        clip.close()
    
    print("\n" + "=" * 60)
    print(f"✅ Video generated successfully!")
    print(f"📁 Output: {output_path}")
    print(f"⏱️  Duration: {sum(clip.duration for _, clip in scenes):.1f} seconds")
    print(f"📺 Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps")
    print("\n🎥 Features showcased:")
    print("   1. Evidence Badges with confidence scoring")
    print("   2. Voice Medication Reminders for elderly")
    print("   3. Warm Handoff via SIP-REFER (Vobiz.ai)")
    print("   4. Human-AI synergistic collaboration")
    print("=" * 60)

if __name__ == "__main__":
    main()
