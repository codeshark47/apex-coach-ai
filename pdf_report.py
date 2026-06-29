import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
# --- TASK 6 IMPORT: BRING IN THE CENTRAL REFERENCE RANGE SYSTEM ---
from reference_ranges import classify_with_range

def generate_automated_pdf(analysis_payload: dict, ai_insights: dict, output_pdf_path: str = "output/Bowling_Analysis_Report.pdf") -> str:
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    meta = analysis_payload.get("video_metadata", {})
    frames = analysis_payload.get("time_indices", {})
    metrics = analysis_payload.get("biomechanical_metrics", {})
    
    source_file = meta.get("source_file", "input_delivery.mp4")
    fps = meta.get("fps", 30.0)
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    PRIMARY = colors.HexColor("#1A365D")
    SECONDARY = colors.HexColor("#2B6CB0")
    TEXT_COLOR = colors.HexColor("#2D3748")
    CRITICAL_COLOR = colors.HexColor("#C53030")
    WARNING_COLOR = colors.HexColor("#DD6B20")
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=22, leading=26, textColor=PRIMARY, spaceAfter=6)
    meta_style = ParagraphStyle('MetaText', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor("#718096"), spaceAfter=15)
    section_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontSize=13, leading=16, textColor=PRIMARY, spaceBefore=14, spaceAfter=8)
    sub_section_style = ParagraphStyle('SubSectionHeader', parent=styles['Heading3'], fontSize=11, leading=14, textColor=SECONDARY, spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontSize=10, leading=14, textColor=TEXT_COLOR)
    
    # Custom inline alerts for table rows
    warn_style = ParagraphStyle('WarningText', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=WARNING_COLOR, leftIndent=10)
    
    story.append(Paragraph("Cricket AI Analyzer - Performance Report", title_style))
    story.append(Paragraph(f"<b>Source Video:</b> {source_file}  |  <b>Frame Rate:</b> {fps} FPS  |  <b>Date:</b> {current_date}", meta_style))
    
    # 1. Timeline Table
    story.append(Paragraph("⏱️ Kinematic Sequence Timeline", section_style))
    timeline_data = [
        [Paragraph("<b>Phase Event</b>", body_style), Paragraph("<b>Detected Frame Number</b>", body_style)],
        [Paragraph("Back Foot Contact (BFC)", body_style), Paragraph(f"Frame {frames.get('back_foot_contact_frame', 'N/A')}", body_style)],
        [Paragraph("Front Foot Contact (FFC)", body_style), Paragraph(f"Frame {frames.get('front_foot_contact_frame', 'N/A')}", body_style)],
        [Paragraph("Ball Release (BR)", body_style), Paragraph(f"Frame {frames.get('ball_release_frame', 'N/A')}", body_style)]
    ]
    t_timeline = Table(timeline_data, colWidths=[260, 260])
    t_timeline.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#E2E8F0")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(t_timeline)
    
    # 2. Upgraded Reference Range Metrics Table
    story.append(Paragraph("📈 Quantified Biomechanical Metrics", section_style))
    
    metrics_data = [
        [
            Paragraph("<b>Metric Measured</b>", body_style), 
            Paragraph("<b>Value</b>", body_style), 
            Paragraph("<b>Zone Status</b>", body_style),
            Paragraph("<b>Reference Threshold Bounds</b>", body_style)
        ]
    ]
    
    # Map out the five core metrics seamlessly
    target_keys = ["front_knee_bracing", "hip_shoulder_separation", "trunk_lean", "release_height", "head_stability"]
    display_names = {
        "front_knee_bracing": "Front Knee Bracing Angle",
        "hip_shoulder_separation": "Hip-Shoulder Separation",
        "trunk_lean": "Lateral Trunk Lean Angle",
        "release_height": "Release Height Leverage Ratio",
        "head_stability": "Head Stability Variance Index"
    }
    
    for key in target_keys:
        item = metrics.get(key)
        name = display_names.get(key, key.replace('_', ' ').title())
        
        if not item:
            metrics_data.append([Paragraph(name, body_style), Paragraph("N/A", body_style), Paragraph("No Data", body_style), Paragraph("N/A", body_style)])
            continue
            
        val = item.get("degrees") or item.get("ratio") or item.get("value") or item.get("deviation_index")
        
        if val is None:
            metrics_data.append([Paragraph(name, body_style), Paragraph("N/A", body_style), Paragraph("Error", body_style), Paragraph("N/A", body_style)])
            continue
            
        # Call the standalone range engine to handle active lab profiling layout rules
        classification = classify_with_range(key, float(val))
        zone_label = classification["zone"].upper()
        
        # Color coordinate the text block zones dynamically
        if zone_label == "CRITICAL":
            zone_html = f"<font color='{CRITICAL_COLOR.hexval()}'><b>CRITICAL</b></font>"
        elif zone_label == "ACCEPTABLE":
            zone_html = f"<font color='{WARNING_COLOR.hexval()}'><b>ACCEPTABLE</b></font>"
        else:
            zone_html = f"<font color='{PRIMARY.hexval()}'><b>OPTIMAL</b></font>"
            
        # Handle string formatting details across units
        unit = classification.get("unit", "")
        unit_str = "°" if unit == "degrees" else ""
        
        val_display = f"<b>{val}{unit_str}</b>"
        range_display = f"Optimal: {classification['optimal_range']} | Acceptable: {classification['acceptable_range']}"
        
        # Add basic metric row entry
        metrics_data.append([
            Paragraph(name, body_style),
            Paragraph(val_display, body_style),
            Paragraph(zone_html, body_style),
            Paragraph(range_display, body_style)
        ])
        
        # Inject warning alert row immediately below if camera anomalies are intercepted
        if classification.get("measurement_warning"):
            metrics_data.append([
                Paragraph(f"⚠️ {classification['measurement_warning']}", warn_style),
                Paragraph("", body_style),
                Paragraph("", body_style),
                Paragraph("", body_style)
            ])

    t_metrics = Table(metrics_data, colWidths=[185, 65, 80, 210])
    
    # Construct base styles, spanning the warning text across columns where needed
    t_styles = [
        ('BACKGROUND', (0, 0), (3, 0), colors.HexColor("#E2E8F0")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]
    
    # Apply column span rules to clear lines for structural warnings
    row_idx = 1
    for key in target_keys:
        item = metrics.get(key)
        if item:
            val = item.get("degrees") or item.get("ratio") or item.get("value") or item.get("deviation_index")
            if val is not None:
                classification = classify_with_range(key, float(val))
                if classification.get("measurement_warning"):
                    row_idx += 1
                    t_styles.append(('SPAN', (0, row_idx), (3, row_idx)))
                    t_styles.append(('BACKGROUND', (0, row_idx), (3, row_idx), colors.HexColor("#FFF5F5")))
        row_idx += 1

    t_metrics.setStyle(TableStyle(t_styles))
    story.append(t_metrics)
    
    # 3. AI Insights Section
    story.append(Paragraph("🧠 Autonomous AI Coach Assessment", section_style))
    story.append(Paragraph("📝 Technical Narrative", sub_section_style))
    
    raw_narrative = ai_insights.get("narrative_analysis", "")
    clean_narrative = raw_narrative.replace("Section 1: Detailed Bio-Mechanical Narrative Review", "").strip()
    for paragraph in clean_narrative.split("\n"):
        if paragraph.strip():
            story.append(Paragraph(paragraph.strip(), body_style))
            story.append(Spacer(1, 6))
            
    story.append(Paragraph("🎯 Prescribed Training Drills", sub_section_style))
    for drill in ai_insights.get("prescribed_drills", []):
        clean_drill = drill.replace("Section 2: Professional Prescribed Drills", "").strip()
        if clean_drill:
            story.append(Paragraph(f"• {clean_drill}", body_style))
            story.append(Spacer(1, 4))
            
    story.append(Spacer(1, 15))
    story.append(Paragraph("<i>Report generated autonomously by Cricket AI Analyzer Core Engine.</i>", body_style))
    story.append(Paragraph("<b>Shoaib Nazar, Founder | Cricket AI Analyzer</b>", body_style))
    
    doc.build(story)
    return output_pdf_path