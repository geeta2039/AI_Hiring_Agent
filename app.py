# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
import json
import tempfile
from dotenv import load_dotenv
import os

st.markdown(
    """
    <style>
    /* Set background color */
    .stApp {
        background-color: #e6f2e6;  /* light green */
    }

    /* Style the main header */
    .main-header {
        color: #006400;  /* dark green */
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }

    /* Style subheaders */
    .sub-header {
        color: #008000;  /* green */
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }

    /* Style file uploader box */
    .stFileUploader > div > label {
        font-weight: bold;
        color: #006400;
    }

    /* Style buttons */
    div.stButton > button {
        background-color: #228B22;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }

    div.stButton > button:hover {
        background-color: #32CD32;  /* lighter green on hover */
        color: white;
    }

    /* Style the report box */
    .report-box {
        background-color: #f0fff0;  /* very light green */
        border-left: 5px solid #006400;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main header
st.markdown('<div class="main-header">Hiring AI Agent</div>', unsafe_allow_html=True)

load_dotenv()
gemini_key = os.getenv("GEMINI_KEY")

# ---------------------------
# Initialize LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=gemini_key)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Hiring Recommendation System")
st.title("AI Hiring Recommendation System")
st.header("Upload Resume & Job Description PDFs")

resume_file = st.file_uploader("Upload Resume PDF", type="pdf")
jd_file = st.file_uploader("Upload Job Description PDF", type="pdf")




# ---------------------------
# Function to load PDF text
def load_pdf_text(uploaded_file):
    # Reset pointer to the start
    uploaded_file.seek(0)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    text = "\n".join([p.page_content for p in pages])
    return text

if resume_file and jd_file:
    # ----------------------------
    # Convert uploaded PDFs to text
    # ----------------------------
    resume_text = load_pdf_text(resume_file)
    job_text = load_pdf_text(jd_file)

    st.success("Resume and Job Description loaded successfully!")

# ---------------------------
# Workflow functions
# ---------------------------
def extract_resume_summary(state: dict) -> dict:
    prompt = PromptTemplate(
        template="""
        Convert the following resume into a VALID JSON object.
        Required JSON keys:
        - skills (list of strings)
        - experience (string)
        - education (string)
        - achievements (string)
        Return ONLY the JSON object.

        Resume:
        {resume_text}
        """,
        input_variables=["resume_text"],
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({"resume_text": state["resume_text"]})

    try:
        resume_dict = json.loads(response)
    except:
        json_part = response[response.find("{"):response.rfind("}") + 1]
        resume_dict = json.loads(json_part)

    state["resume_summary"] = resume_dict
    return state

def extract_job_summary(state: dict) -> dict:
    prompt = PromptTemplate(
        template="""
        Convert the Job Description below into a VALID JSON object.
        JSON keys must be:
        - required_skills (list of strings)
        - experience_level (string)
        - education_requirements (string)
        - responsibilities (string)
        Return ONLY JSON.

        Job Description:
        {job_text}
        """,
        input_variables=["job_text"],
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({"job_text": state["job_text"]})

    try:
        job_dict = json.loads(response)
    except:
        job_dict = json.loads(response[response.find("{"):response.rfind("}") + 1])

    state["job_summary"] = job_dict
    return state

def compare_and_score(state: dict) -> dict:
    resume = state["resume_summary"]
    job = state["job_summary"]

    def compute_similarity(a, b):
        if not a or not b:
            return 0
        if isinstance(a, list):
            a = " ".join(a)
        if isinstance(b, list):
            b = " ".join(b)
        a = a.lower()
        b = b.lower()
        return len(set(a.split()) & set(b.split()))

    skill_score = compute_similarity(resume["skills"], job["required_skills"]) * 10
    exp_score = compute_similarity(resume["experience"], job["experience_level"]) * 10
    edu_score = compute_similarity(resume["education"], job["education_requirements"]) * 10

    total = round((skill_score*0.5) + (exp_score*0.3) + (edu_score*0.2), 2)

    state["scores"] = {"skills": skill_score, "experience": exp_score, "education": edu_score}
    state["score"] = total
    return state

def decide_interview_stage(state: dict) -> dict:
    score = state["score"]
    if score >= 85:
        state["decision"] = "One Interview"
    elif score >= 60:
        state["decision"] = "Two Interviews"
    else:
        state["decision"] = "Rejected"
    return state

def one_interview(state: dict) -> dict:
    state["process"] = "Direct human interview scheduled"
    return state

def two_interviews(state: dict) -> dict:
    state["process"] = "Screening + Coding round scheduled"
    return state

def reject_process(state: dict) -> dict:
    state["process"] = "Candidate rejected. Regret email triggered."
    return state

def generate_report(state: dict) -> dict:
    scores = state["scores"]
    total = state["score"]
    decision = state["decision"]
    report = f"""
📋 FINAL HIRING RECOMMENDATION REPORT
-------------------------------------
Skills Score: {scores['skills']}
Experience Score: {scores['experience']}
Education Score: {scores['education']}
Overall Score: {total}

Recommendation: {decision}

Summary:
Candidate demonstrates measurable alignment with job requirements.
"""
    state["final_report"] = report
    return state

# ---------------------------
# Run workflow if files uploaded
# ---------------------------
if resume_file and jd_file:
    resume_text = load_pdf_text(resume_file)
    job_text = load_pdf_text(jd_file)

    graph = StateGraph(dict)
    graph.add_node("resume_summary", extract_resume_summary)
    graph.add_node("job_summary", extract_job_summary)
    graph.add_node("scoring", compare_and_score)
    graph.add_node("decision", decide_interview_stage)
    graph.add_node("one", one_interview)
    graph.add_node("two", two_interviews)
    graph.add_node("reject", reject_process)
    graph.add_node("report", generate_report)

    graph.add_edge(START, "resume_summary")
    graph.add_edge("resume_summary", "job_summary")
    graph.add_edge("job_summary", "scoring")
    graph.add_edge("scoring", "decision")

    graph.add_conditional_edges(
        "decision",
        lambda state: state["decision"],
        {
            "One Interview": "one",
            "Two Interviews": "two",
            "Rejected": "reject",
        },
    )

    graph.add_edge("one", "report")
    graph.add_edge("two", "report")
    graph.add_edge("reject", "report")
    graph.add_edge("report", END)

    workflow = graph.compile()
    state = workflow.invoke({"resume_text": resume_text, "job_text": job_text})

    st.subheader("Final Report")
    st.text(state["final_report"])
else:
    st.info("Please upload both Resume and Job Description PDFs to continue.")