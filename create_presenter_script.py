#!/usr/bin/env python3
"""
Presenter Script Generator for Palli Sahayak Workshop Presentation
Creates a PDF script with speaker notes, timing, and engagement cues.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import os

# Colors
PRIMARY = HexColor('#2E7D32')
SECONDARY = HexColor('#00696B')
ACCENT = HexColor('#FF8F00')
DARK = HexColor('#212121')
LIGHT_GRAY = HexColor('#F5F5F5')
TAG_BLUE = HexColor('#1565C0')
TAG_PURPLE = HexColor('#6A1B9A')

def create_styles():
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='SlideTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=PRIMARY,
        spaceAfter=6,
        spaceBefore=12,
    ))

    styles.add(ParagraphStyle(
        name='SlideNum',
        parent=styles['Normal'],
        fontSize=11,
        textColor=SECONDARY,
        spaceBefore=20,
    ))

    styles.add(ParagraphStyle(
        name='Script',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name='Tag',
        parent=styles['Normal'],
        fontSize=10,
        textColor=TAG_PURPLE,
        spaceBefore=4,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name='Timing',
        parent=styles['Normal'],
        fontSize=9,
        textColor=ACCENT,
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=SECONDARY,
        spaceBefore=20,
        spaceAfter=12,
        alignment=TA_CENTER,
    ))

    return styles

# Speaker script content for each slide
SCRIPT_CONTENT = [
    # Slide 1: Title
    {
        "slide_num": 1,
        "title": "Title Slide",
        "duration": "30 sec",
        "script": """
<b><font color="#6A1B9A">&lt;HOOK - Start with a question&gt;</font></b>

"Imagine it's 2 AM. Your mother is in pain. Cancer pain. You don't know what to do. The doctor's clinic is closed. Google gives you 50 contradicting answers. <b><font color="#1565C0">&lt;pause - 3 sec&gt;</font></b>

What do you do?"

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

"This is the reality for over 10 million Indian families RIGHT NOW. Today, I'm going to show you how we're changing that."

<i>Walk to center stage, make eye contact</i>
""",
    },

    # Slide 2: Acknowledgements
    {
        "slide_num": 2,
        "title": "Acknowledgements",
        "duration": "45 sec",
        "script": """
"Before we dive in, I want to acknowledge the shoulders we stand on.

This work is possible because of the Bill & Melinda Gates Foundation and BIRAC through Grand Challenges India. <b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

Our PI, Dr. Anurag Agrawal - Dean of Trivedi School of Biosciences and Head of KCDH-Ashoka - whose vision drives this project.

And our incredible clinical partners: Pallium India and IAPC, who ensure everything we build actually WORKS in the real world."

<i>Nod to acknowledge, brief but sincere</i>
""",
    },

    # Slide 3: The Crisis
    {
        "slide_num": 3,
        "title": "The Palliative Care Crisis",
        "duration": "50 sec",
        "script": """
<b><font color="#6A1B9A">&lt;Poll the audience&gt;</font></b>

"Quick show of hands - how many of you have had a family member who needed palliative care?"

<b><font color="#1565C0">&lt;pause - 3 sec, scan the room&gt;</font></b>

"Now, let these numbers sink in:

10 MILLION Indians need palliative care. Only 1-2% receive it. <b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

That's not a gap. That's a CHASM.

We have 0.4 palliative care specialists per 100,000 people. The US has 3. The UK has 5.

But here's the thing - we can't train doctors fast enough. So what's the alternative?"

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>
""",
    },

    # Slide 4: India's Digital Opportunity
    {
        "slide_num": 4,
        "title": "India's Digital Opportunity",
        "duration": "40 sec",
        "script": """
"Here's where India has a secret superpower.

<b><font color="#6A1B9A">&lt;Can someone tell me how many active mobile connections India has?&gt;</font></b>

<b><font color="#1565C0">&lt;pause - 3 sec&gt;</font></b>

1.2 BILLION. More phones than toilets, as they say. <i>(slight smile)</i>

And here's the beautiful part - voice calls work EVERYWHERE. No 4G needed. No smartphone needed. No literacy needed.

The phone is already in every hand. We just need to make it smarter."
""",
    },

    # Slide 5: Introducing Palli Sahayak
    {
        "slide_num": 5,
        "title": "Introducing Palli Sahayak",
        "duration": "45 sec",
        "script": """
"So we built Palli Sahayak - which means 'Companion in Care' in Hindi.

It's NOT a chatbot. <b><font color="#1565C0">&lt;pause - 1 sec&gt;</font></b> It's NOT a symptom checker. <b><font color="#1565C0">&lt;pause - 1 sec&gt;</font></b>

It's a voice-first AI helpline that speaks YOUR language - literally 15+ Indian languages - available 24/7, answering the question: 'What do I do RIGHT NOW?'

<b><font color="#6A1B9A">&lt;Ask a q&gt;</font></b> What makes this different from Dr. Google? <b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

Clinical validation. Every response is grounded in palliative care guidelines, checked for safety, verified by experts. This isn't generic health info - it's ACTIONABLE guidance."
""",
    },

    # Slide 6: System Architecture
    {
        "slide_num": 6,
        "title": "System Architecture",
        "duration": "40 sec",
        "script": """
"Let me show you what's under the hood - but I promise to keep it simple.

<i>Point to diagram</i>

You call or WhatsApp. Your voice goes through one of three swappable AI providers - we'll see those next. Gets transcribed, processed through our RAG pipeline - that's where the clinical knowledge lives - and comes back as natural speech.

The magic? Under 1 second latency. It feels like talking to a person, not waiting for a computer.

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

Now let me show you the REALLY cool part..."
""",
    },

    # Slide 7: LIVE DEMO slide
    {
        "slide_num": 7,
        "title": "LIVE DEMO - Key Innovations",
        "duration": "30 sec + 10 min demo",
        "script": """
"Alright, enough slides. Let's see this thing in action."

<b><font color="#6A1B9A">&lt;DEMO TIME - 10 minutes&gt;</font></b>

<i>Demo sequence:</i>
1. Start with Hindi voice call - "Meri maa ko bahut dard ho raha hai" (2 min)
2. Switch to English WhatsApp - medication question (2 min)
3. Show Hinglish interaction - code-switching (2 min)
4. Demonstrate safety guardrails - show what happens with emergency query (2 min)
5. Show admin dashboard - real-time analytics (2 min)

<b><font color="#1565C0">&lt;pause after demo - 3 sec&gt;</font></b>

"Questions so far before we continue?"
""",
    },

    # Slide 8: Voice AI Providers
    {
        "slide_num": 8,
        "title": "Voice AI Providers - Swappable Architecture",
        "duration": "50 sec",
        "script": """
"Here's something I'm particularly proud of - our provider-agnostic architecture.

We don't lock into one vendor. Three providers, completely swappable:

<i>Point to each</i>

Gemini Live - Google's native audio, 4 Indian languages, incredibly natural.

Bolna.ai - 5 languages including Hinglish - that beautiful mix of Hindi and English that 400 million Indians actually speak.

Retell.ai with Vobiz - real Indian phone numbers, +91, works on any phone.

<b><font color="#6A1B9A">&lt;Can someone tell me why having multiple providers matters?&gt;</font></b>

<b><font color="#1565C0">&lt;pause - 3 sec&gt;</font></b>

Exactly - redundancy, cost optimization, and no vendor lock-in. This is how you build infrastructure that LASTS."
""",
    },

    # Slide 9: Voice-First Design
    {
        "slide_num": 9,
        "title": "Why Voice-First Design",
        "duration": "35 sec",
        "script": """
"Why voice? <b><font color="#1565C0">&lt;pause - 1 sec&gt;</font></b>

Because dignity means being understood in your mother tongue.

<b><font color="#6A1B9A">&lt;Ask a q&gt;</font></b> How many of your parents are comfortable typing in English on a smartphone?

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

Voice is universal. Voice is natural. Voice works at 2 AM when you're exhausted and scared.

No app download. No login. No typing. Just... talk."
""",
    },

    # Slide 10: Languages
    {
        "slide_num": 10,
        "title": "15+ Indian Languages",
        "duration": "30 sec",
        "script": """
"Hindi, Tamil, Telugu, Bengali, Marathi, Kannada... and yes, Hinglish.

<i>Smile</i>

Because let's be honest - most of urban India doesn't speak 'pure' Hindi or 'pure' English. We code-switch constantly. 'Mujhe morning mein pain hota hai' - that's real speech.

Our system understands that."
""",
    },

    # Slide 11: Clinical RAG Foundation
    {
        "slide_num": 11,
        "title": "Clinical RAG Foundation",
        "duration": "40 sec",
        "script": """
"Now, the brain of the system - our Clinical RAG.

RAG stands for Retrieval Augmented Generation. Fancy words, simple concept:

Instead of the AI making things up, it RETRIEVES real information from curated palliative care knowledge, THEN generates a response.

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

We're talking WHO guidelines, IAPC protocols, Pallium India resources - over 200 verified documents. Not Reddit posts. Not random blogs."
""",
    },

    # Slide 12: Triple-Layer Architecture
    {
        "slide_num": 12,
        "title": "Triple-Layer Knowledge Architecture",
        "duration": "45 sec",
        "script": """
"Here's where it gets interesting. We don't just do simple search.

Three layers: <i>count on fingers</i>

ONE - Vector search. Semantic similarity. 'Pain' matches 'discomfort' matches 'it hurts'.

TWO - Knowledge Graph. Relationships. 'Morphine treats pain' but 'Morphine requires prescription' and 'Morphine has side effects'.

THREE - GraphRAG. Community detection. Understanding CONTEXT across multiple documents.

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

Then we FUSE these layers intelligently. That's the secret sauce."
""",
    },

    # Slide 13: Clinical Validation
    {
        "slide_num": 13,
        "title": "Clinical Validation Pipeline",
        "duration": "40 sec",
        "script": """
<b><font color="#6A1B9A">&lt;Poll the audience&gt;</font></b>

"Who here trusts AI to give medical advice without human oversight?"

<b><font color="#1565C0">&lt;pause - 3 sec, expect few hands&gt;</font></b>

"Good. Neither do we.

Every response goes through: dosage verification, drug interaction checks, emergency detection, and periodic expert sampling.

The AI doesn't replace doctors. It EXTENDS their reach to 2 AM, to remote villages, to overwhelmed caregivers."
""",
    },

    # Slide 14: Personalization
    {
        "slide_num": 14,
        "title": "Adaptive User Experience",
        "duration": "30 sec",
        "script": """
"The system learns. Not in a creepy way - in a helpful way.

It remembers: this patient is on morphine. This caregiver prefers Hindi. This family has asked about anxiety before.

Context persistence. So you don't repeat your story every single time."
""",
    },

    # Slide 15: Analytics
    {
        "slide_num": 15,
        "title": "Real-time Analytics Dashboard",
        "duration": "30 sec",
        "script": """
"And for program managers - real-time visibility.

What are people asking about? Where are the pain points - literally? Which regions need more support?

Data-driven palliative care. Novel concept, right?"

<i>Brief smile</i>
""",
    },

    # Slide 16: DPG
    {
        "slide_num": 16,
        "title": "Digital Public Good (DPG)",
        "duration": "50 sec",
        "script": """
"Now, here's the BIGGER picture.

<b><font color="#6A1B9A">&lt;Ask a q&gt;</font></b> What is a Digital Public Good?

<b><font color="#1565C0">&lt;pause - 3 sec&gt;</font></b>

Open source. Open standards. Designed for replication.

Palli Sahayak isn't just a project - it's INFRASTRUCTURE. Like Aadhaar. Like UPI. But for healthcare.

Any government, any NGO, any healthcare system can take this, adapt it, deploy it. The code is on GitHub. The docs are public. The knowledge is shared."
""",
    },

    # Slide 17: Global Replicability
    {
        "slide_num": 17,
        "title": "Global Replicability",
        "duration": "45 sec",
        "script": """
"And this isn't just for India.

<b><font color="#6A1B9A">&lt;Can someone tell me which developed country has perfect palliative care coverage?&gt;</font></b>

<b><font color="#1565C0">&lt;pause - 3 sec&gt;</font></b>

None. The US has 1 million people who need hospice but don't get it. UK has rural access issues. Japan has an aging crisis.

This architecture - voice-first, multilingual, clinically validated - works ANYWHERE. Replace Hindi with Spanish, replace IAPC guidelines with NHPCO guidelines. The bones are the same."
""",
    },

    # Slide 18: Impact Metrics
    {
        "slide_num": 18,
        "title": "Impact Metrics & Targets",
        "duration": "35 sec",
        "script": """
"What does success look like?

Pilot: 1,000 families. Measure: symptom resolution time, caregiver stress, unnecessary ER visits.

Scale: 100,000 families in 24 months.

Moon shot: Integrate with National Health Mission. Make this the palliative care backbone for Ayushman Bharat.

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

Ambitious? Yes. Impossible? Watch us."
""",
    },

    # Slide 19: Partners
    {
        "slide_num": 19,
        "title": "Partners & Collaborations",
        "duration": "30 sec",
        "script": """
"We don't build alone.

Pallium India - clinical expertise, training data, real-world validation.

IAPC - the Indian Association of Palliative Care - guidelines, network, credibility.

This is what responsible AI development looks like - technologists and clinicians, together."
""",
    },

    # Slide 20: Tech Stack
    {
        "slide_num": 20,
        "title": "Technology Stack",
        "duration": "30 sec",
        "script": """
"For the techies in the room - yes, it's Python. Yes, it's FastAPI. Yes, it's open source.

ChromaDB for vectors, Neo4j for knowledge graphs, multiple LLM backends.

The stack is boring on purpose. Boring means maintainable. Boring means any developer can contribute."
""",
    },

    # Slide 21: Indian Context
    {
        "slide_num": 21,
        "title": "Indian Socio-Cultural Context",
        "duration": "40 sec",
        "script": """
"But technology is only half the story.

We designed for INDIA. Joint families where decisions are collective. Stigma around morphine. Regional dietary restrictions during illness. Spiritual needs alongside medical ones.

<b><font color="#6A1B9A">&lt;Ask a q&gt;</font></b> How many medical AI systems consider that a Hindu family might need guidance on rituals during end-of-life?

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

Ours does."
""",
    },

    # Slide 22: Roadmap
    {
        "slide_num": 22,
        "title": "Development Roadmap",
        "duration": "30 sec",
        "script": """
"Where are we going?

Next: Pilot in 3 states. Then: Integration with telemedicine platforms. Eventually: A national helpline number that anyone can call.

The roadmap is aggressive because the need is urgent."
""",
    },

    # Slide 23: Call to Action
    {
        "slide_num": 23,
        "title": "Call to Action",
        "duration": "45 sec",
        "script": """
<b><font color="#6A1B9A">&lt;CLOSING - Bring energy back up&gt;</font></b>

"So what can YOU do?

If you're a clinician - help us validate. Your expertise makes this safe.

If you're a researcher - collaborate. There's a dozen papers waiting to be written.

If you're a funder - invest. Not in a startup, in INFRASTRUCTURE.

If you're anyone else - spread the word. Someone you know needs this."

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>
""",
    },

    # Slide 24: Thank You
    {
        "slide_num": 24,
        "title": "Thank You",
        "duration": "30 sec + 5 min Q&A",
        "script": """
<b><font color="#6A1B9A">&lt;CLOSING HOOK - Circle back to opening&gt;</font></b>

"Remember that 2 AM scenario I started with?

<b><font color="#1565C0">&lt;pause - 2 sec&gt;</font></b>

Now imagine: the caregiver picks up the phone, speaks in Hindi, gets clear guidance on managing the pain, knows when to call the doctor, and gets through the night.

THAT'S what we're building.

Thank you. I'll take questions."

<b><font color="#6A1B9A">&lt;Q&A - 5 minutes&gt;</font></b>
""",
    },
]

def create_script_pdf(output_path):
    """Generate the presenter script PDF."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = create_styles()
    story = []

    # Title page
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph(
        "<b>PALLI SAHAYAK</b>",
        ParagraphStyle(
            name='MainTitle',
            parent=styles['Title'],
            fontSize=28,
            textColor=PRIMARY,
            alignment=TA_CENTER,
        )
    ))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        "Presenter Script & Speaker Notes",
        ParagraphStyle(
            name='Subtitle',
            parent=styles['Normal'],
            fontSize=18,
            textColor=SECONDARY,
            alignment=TA_CENTER,
        )
    ))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "Workshop Presentation - January 2026",
        ParagraphStyle(
            name='Date',
            parent=styles['Normal'],
            fontSize=12,
            textColor=DARK,
            alignment=TA_CENTER,
        )
    ))

    story.append(Spacer(1, 1*inch))

    # Timing summary
    timing_data = [
        ["Section", "Duration"],
        ["Slides 1-6 (Opening & Context)", "~4 min"],
        ["Slide 7 (LIVE DEMO)", "~10 min"],
        ["Slides 8-15 (Technical Deep Dive)", "~5 min"],
        ["Slides 16-22 (Vision & Roadmap)", "~4 min"],
        ["Slides 23-24 (CTA & Close)", "~2 min"],
        ["Q&A", "5 min"],
        ["TOTAL", "30 min"],
    ]

    timing_table = Table(timing_data, colWidths=[3.5*inch, 1.5*inch])
    timing_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, -1), (-1, -1), LIGHT_GRAY),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
    ]))
    story.append(timing_table)

    story.append(Spacer(1, 0.5*inch))

    # Legend
    story.append(Paragraph("<b>TAG LEGEND:</b>", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    legend_text = """
    <font color="#1565C0">&lt;pause - X sec&gt;</font> = Strategic pause for emphasis<br/>
    <font color="#6A1B9A">&lt;Ask a q&gt;</font> = Rhetorical question to audience<br/>
    <font color="#6A1B9A">&lt;Poll the audience&gt;</font> = Request show of hands<br/>
    <font color="#6A1B9A">&lt;Can someone tell me...&gt;</font> = Engage specific audience member<br/>
    <i>Italics</i> = Stage directions / body language cues
    """
    story.append(Paragraph(legend_text, styles['Script']))

    story.append(PageBreak())

    # Script content for each slide
    story.append(Paragraph("PRESENTER SCRIPT", styles['SectionHeader']))
    story.append(Spacer(1, 0.2*inch))

    for slide in SCRIPT_CONTENT:
        # Slide header
        story.append(Paragraph(
            f"SLIDE {slide['slide_num']}: {slide['title']}",
            styles['SlideNum']
        ))
        story.append(Paragraph(
            f"<font color='#FF8F00'>[Duration: {slide['duration']}]</font>",
            styles['Timing']
        ))
        story.append(Spacer(1, 0.1*inch))

        # Script content
        script_text = slide['script'].strip().replace('\n', '<br/>')
        story.append(Paragraph(script_text, styles['Script']))
        story.append(Spacer(1, 0.2*inch))

    # Build PDF
    doc.build(story)
    print(f"Created presenter script: {output_path}")

if __name__ == "__main__":
    output_dir = "/Users/tp53/Documents/tp53_AA/llms4palliative_gci/demo_feb2025/rag_gci/docs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo_script.pdf")
    create_script_pdf(output_path)
