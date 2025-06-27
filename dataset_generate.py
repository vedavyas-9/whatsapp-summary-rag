import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from docx import Document
from docx.shared import Pt
import json
from datetime import datetime, timedelta
import random
import uuid
import os

# Function to generate random Indian phone number
def random_indian_phone():
    prefixes = ['7', '8', '9']
    return f"+91-{random.choice(prefixes)}{''.join([str(random.randint(0, 9)) for _ in range(9)])}"

# Function to generate random timestamp within a date
def random_timestamp(date, start_hour=8, end_hour=20):
    hour = random.randint(start_hour, end_hour)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime.strptime(f"{date} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")

# Function to generate synthetic IPDR data (20–30 entries)
def generate_ipdr_data(subscribers, date, set_num):
    data = []
    num_entries = random.randint(20, 30)
    for _ in range(num_entries):
        subscriber = random.choice(subscribers)
        session_start = random_timestamp(date)
        session_end = session_start + timedelta(minutes=random.randint(15, 120))
        bytes_transferred = random.randint(500000, 15000000)  # 0.5MB to 15MB
        apn = random.choice(["apn.mobile.net", "apn.data.net"])
        data.append([subscriber, session_start.strftime("%Y-%m-%d %H:%M:%S"),
                     session_end.strftime("%Y-%m-%d %H:%M:%S"), bytes_transferred, apn])
    return [["subscriberID", "sessionStart", "sessionEnd", "bytesTransferred", "APN"]] + data

# Function to generate synthetic CDR data (50–60 entries)
def generate_cdr_data(subscribers, date, set_num):
    data = []
    towers = [f"TWR00{i}" for i in range(set_num, set_num + 5)]  # More towers for variety
    num_entries = random.randint(50, 60)
    for _ in range(num_entries):
        caller = random.choice(subscribers)
        callee = random.choice(subscribers + [random_indian_phone() for _ in range(3)])
        start_time = random_timestamp(date)
        end_time = start_time + timedelta(minutes=random.randint(5, 20))
        tower = random.choice(towers)
        data.append([caller, callee, start_time.strftime("%Y-%m-%d %H:%M:%S"),
                     end_time.strftime("%Y-%m-%d %H:%M:%S"), tower])
    return pd.DataFrame(data, columns=["callerID", "calleeID", "startTime", "endTime", "towerID"])

# Function to generate synthetic FIR data (1–2 pages)
def generate_fir_data(subscribers, date, set_num, cdr_data):
    case_id = f"CASE00{set_num}"
    crime_type = random.choice(['Theft', 'Cybercrime', 'Smuggling', 'Fraud'])
    location = random.choice(['Vijayawada Market', 'Port Area', 'City Center', 'Guntur Road'])
    suspect = random.choice(subscribers)
    related_phone = random.choice(cdr_data['calleeID'].tolist())
    tower = random.choice(cdr_data['towerID'].tolist())
    time_window = cdr_data['startTime'].iloc[0][:16]

    # Detailed FIR narrative (simulating LLM output)
    narrative = f"""
First Information Report (FIR)
Andhra Pradesh Police Department

Case ID: {case_id}
Incident Date: {date}
Location of Incident: {location}
Crime Type: {crime_type}

1. Incident Details:
On {date}, at approximately {time_window}, a {crime_type.lower()} was reported at {location}. The incident involved suspicious activities linked to suspect with phone number {suspect}. The suspect was observed engaging in communication with another individual ({related_phone}) during the time of the incident. The activities were traced to cellular tower {tower}, indicating the suspect's presence in the vicinity.

2. Suspect Information:
- Primary Suspect: {suspect}
- Known Associates: The suspect was in contact with {related_phone}, who is under investigation for potential involvement.
- Modus Operandi: {random.choice(['The suspect used encrypted communication channels.', 'The suspect coordinated via multiple calls.', 'The suspect transferred large data volumes.'])}
- Additional Notes: The suspect's movements were tracked through tower handoffs, suggesting a pattern of activity across multiple locations.

3. Witness Statements:
A local shopkeeper reported seeing an individual matching the suspect's description near {location} at the time of the incident. The witness noted suspicious behavior, including frequent phone usage and hurried movements. Another witness, a passerby, corroborated the presence of a vehicle near the scene, which is under further investigation.

4. Preliminary Investigation:
Initial analysis of call detail records (CDR) indicates {random.randint(5, 15)} calls made by {suspect} on {date}, with a significant number routed through tower {tower}. Internet Protocol Detail Records (IPDR) show high data usage, potentially linked to {random.choice(['data exfiltration', 'streaming illicit content', 'coordinating with accomplices'])}. The investigation team has seized digital evidence for forensic analysis.

5. Officer Remarks:
The case is classified as high priority due to its potential impact on public safety. The investigation team is directed to analyze all communication logs and cross-reference with other active cases. Coordination with cybercrime units is recommended for cases involving data breaches.

Location Keywords: {location}, {tower}
Reporting Officer: Inspector A. Sharma
Date Filed: {date}

[Investigation Ongoing]
"""
    return narrative

# Function to create PDF file
def create_pdf(data, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    doc.build([table])

# Function to create Word file with formatting
def create_word(narrative, filename):
    doc = Document()
    # Set document style
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    # Add narrative with paragraphs
    for paragraph in narrative.split('\n'):
        if paragraph.strip():
            doc.add_paragraph(paragraph.strip())
    doc.save(filename)

# Function to create metadata
def create_metadata(set_num, file_type, filename, definition):
    return {
        "Name": filename.split('.')[0],
        "Type": file_type,
        "Definition": definition,
        "Intelligence Level": 1,
        "Format": filename.split('.')[-1].upper()
    }

# Main function to generate datasets
def generate_datasets(num_sets):
    subscribers = [f"+91-{random_indian_phone()[4:]}" for _ in range(7)]  # Indian phone numbers
    base_date = datetime(2025, 6, 20)
    metadata_list = []

    # Create output directory
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    for set_num in range(1, num_sets + 1):
        date = (base_date + timedelta(days=set_num - 1)).strftime("%Y-%m-%d")

        # Generate IPDR (PDF)
        ipdr_data = generate_ipdr_data(subscribers, date, set_num)
        ipdr_filename = f"datasets/IPDR_Set{set_num}.pdf"
        create_pdf(ipdr_data, ipdr_filename)
        metadata_list.append(create_metadata(set_num, "IPDR", ipdr_filename,
                                            "Internet session logs for subscribers"))

        # Generate CDR (Excel)
        cdr_data = generate_cdr_data(subscribers, date, set_num)
        cdr_filename = f"datasets/CDR_Set{set_num}.xlsx"
        cdr_data.to_excel(cdr_filename, index=False)
        metadata_list.append(create_metadata(set_num, "CDR", cdr_filename,
                                            "Call detail records with caller and callee information"))

        # Generate FIR (Word)
        fir_narrative = generate_fir_data(subscribers, date, set_num, cdr_data)
        fir_filename = f"datasets/FIR_Set{set_num}.docx"
        create_word(fir_narrative, fir_filename)
        metadata_list.append(create_metadata(set_num, "FIR", fir_filename,
                                            "First Information Report detailing an incident"))

    # Save metadata to JSON
    with open("datasets/metadata.json", "w") as f:
        json.dump(metadata_list, f, indent=4)

    print(f"Generated {num_sets} sets of datasets (PDF, Excel, Word) in 'datasets' folder.")
    print("Metadata saved to 'datasets/metadata.json'.")

# Prompt user for number of sets
if __name__ == "__main__":
    try:
        num_sets = int(input("Enter the number of dataset sets to generate: "))
        if num_sets < 1:
            print("Please enter a positive number.")
        else:
            generate_datasets(num_sets)
    except ValueError:
        print("Please enter a valid integer.")