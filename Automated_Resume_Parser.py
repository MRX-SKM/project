import spacy
import re
from pdfminer.high_level import extract_text

nlp = spacy.load("en_core_web_sm")

def extract_resume_text(pdf_path):
    return extract_text(pdf_path)

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_skills(text, skill_set):
    return [skill for skill in skill_set if skill.lower() in text.lower()]

text = extract_resume_text("resume.pdf")
print("Name:", extract_name(text))
print("Skills:", extract_skills(text, ["Python", "Django", "TensorFlow"]))

