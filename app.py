from faster_whisper import WhisperModel
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from datetime import datetime, date, timedelta
import datefinder

st.title("Audio Transcription App")

whisper_model = st.selectbox("Select Whisper Model", ["tiny", "base", "small"], index=0)

compute_type = st.selectbox("Select Compute Size", ["int8", "float32"], index=0)

model = WhisperModel(f"{whisper_model}.en", device="cpu", compute_type=compute_type)

with st.sidebar:
    st.header("LLM Configuration")
    select_llm = st.selectbox(
        'Choose the LLM you want to use',
        ('OpenAI', 'Gemini'),
        index=None,
        placeholder='Select the LLM...',
    )
    
    if select_llm == 'OpenAI':
        openai_api_key = st.text_input("Enter your OpenAI API Key:", type='password', key='openai_api_key')
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            st.session_state['OPENAI_API_KEY'] = openai_api_key
            st.success("OpenAI API Key set successfully.")
            llm = ChatOpenAI(model_name="gpt-4o") 
            st.session_state['llm'] = llm

    elif select_llm == 'Gemini':
        google_api_key = st.text_input("Enter your Google API Key:", type='password', key='google_api_key')
        if google_api_key:
            os.environ['GOOGLE_API_KEY'] = google_api_key
            st.session_state['GOOGLE_API_KEY'] = google_api_key
            st.success("Google API Key set successfully.")
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            st.session_state['llm'] = llm

class TaskAssignment(BaseModel):
    name: Optional[str] = Field(default=None)
    task: Optional[str] = Field(default=None)
    department: Optional[str] = Field(default=None)
    start_date: Optional[str] = Field(default=None)
    end_date: Optional[str] = Field(default=None)

class TaskInfo(BaseModel):
    assignments: List[TaskAssignment] = []

def process_transcription(transcription):
    if 'llm' not in st.session_state:
        st.error("Please configure the LLM first")
        return

    try:
        today = date.today()
        tomorrow = today + timedelta(days=1)
        dates_found = list(datefinder.find_dates(transcription))
        
        valid_dates = [d.date() for d in dates_found if d.date() >= today]
        valid_dates.sort()  

        date_context = ""
        if valid_dates:
            date_context = f"Dates found in transcription: {', '.join([d.strftime('%Y-%m-%d') for d in valid_dates])}. "
            date_context += "Use these dates for start_date and end_date where appropriate. "
        else:
            date_context = "No explicit dates found in transcription. Use today for start_date and tomorrow for end_date. "

        structured_llm = st.session_state['llm'].with_structured_output(TaskInfo)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Extract the following fields from the text into a structured format:
                        - Name: Return the person's name if mentioned, otherwise return None.
                        - Task: Extract the assigned task or activity. If not available, return None.
                        - Department: Extract the department or team if specified. If missing, return None.
                        - Start Date & End Date:
                        - Always return both dates in YYYY-MM-DD format. Never return None.
                        - {date_context}
                        - For phrases like 'by tomorrow' or 'due next Monday,' set end_date to the mentioned date and start_date to today, unless specified otherwise.
                        - If a single date is mentioned (e.g., 'on 2025-04-20'), use it for both start_date and end_date unless context indicates it's a deadline.
                        - Ensure end_date is not earlier than start_date; if it is, set end_date to start_date.
                """),
            ("human", "{text}")
        ])
        extraction_chain = prompt | structured_llm
        extracted = extraction_chain.invoke({"text": transcription})
        
        assignments = []

        for task in extracted.assignments:
            start_date = datetime.strptime(task.start_date, "%Y-%m-%d").date() if task.start_date else today
            end_date = datetime.strptime(task.end_date, "%Y-%m-%d").date() if task.end_date else tomorrow

            if end_date < start_date:
                end_date = start_date

            assignments.append({
                "name": task.name,
                "task": task.task,
                "department": task.department,
                "start_date": start_date,
                "end_date": end_date
            })

        st.subheader("Extracted Tasks:")
        if not assignments:
            st.warning("No tasks found")
            return
        
        data = [{
            "Name": str(assignment["name"]) if assignment["name"] else "",
            "Task": str(assignment["task"]) if assignment["task"] else "",
            "Department": str(assignment["department"]) if assignment["department"] else "",
            "Start Date": str(assignment["start_date"]),
            "End Date": str(assignment["end_date"])
        } for assignment in assignments]
        
        st.table(data)

    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")

if 'llm' in st.session_state:
    option = st.radio("Choose an option:", ("Upload an audio file", "Record audio"))

    if option == "Upload an audio file":
        audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "m4a"])
        if audio_file:
            st.audio(audio_file)
            if st.button("Transcribe"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(audio_file.getbuffer())
                    tmp_path = tmp_file.name
                
                segments, info = model.transcribe(tmp_path)
                transcription = " ".join([seg.text for seg in segments])
                
                st.subheader("Transcription:")
                st.write(transcription)
                
                process_transcription(transcription)
                os.unlink(tmp_path)

    elif option == "Record audio":
        audio_bytes = st.audio_input("Record your audio")
        if audio_bytes:
            st.audio(audio_bytes)
            if st.button("Transcribe"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes.getvalue())
                    tmp_path = tmp_file.name

                segments, info = model.transcribe(tmp_path)
                transcription = " ".join([seg.text for seg in segments])

                st.subheader("Transcription:")
                st.write(transcription)

                process_transcription(transcription)
                os.unlink(tmp_path)
else:
    st.warning("Please configure your LLM in the sidebar to use the app.")
    