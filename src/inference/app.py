import streamlit as st
from openai import OpenAI
import requests
import pandas as pd
import json
from typing import Dict, List
import altair as alt
from os import getenv

client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="Toxic Comment Classifier & ChatBot",
    page_icon="🤖",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

def check_toxicity(text: str) -> Dict[str, float]:
    """Check text toxicity using our API"""
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": text}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error checking toxicity: {str(e)}")
        return {}

def format_toxicity_results(results: Dict[str, float]) -> str:
    """Format toxicity results for display"""
    return ", ".join([f"{k}: {v:.2%}" for k, v in results.items()])

def display_toxicity_metrics(results: Dict[str, float]):
    """Display toxicity metrics as a bar chart"""
    if not results:
        return
    
    df = pd.DataFrame({
        'Category': list(results.keys()),
        'Score': list(results.values())
    })
    
    # Create bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1]), title='Probability Score'),
        y=alt.Y('Category:N', sort='-x', title='Toxicity Category'),
        color=alt.Color('Score:Q', 
            scale=alt.Scale(domain=[0, 1], scheme='redyellowgreen', reverse=True),
            legend=alt.Legend(title='Probability')
        )
    ).properties(
        title='Toxicity Scores',
        width=600
    )
    
    st.altair_chart(chart)

def chatbot_tab():
    st.header("💬 ChatGPT with Toxicity Moderation")
    

    if not getenv("OPENAI_API_KEY"):
        st.warning("Please set the OPENAI_API_KEY environment variable.")
        return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "toxicity" in message:
                with st.expander("View toxicity analysis"):
                    display_toxicity_metrics(message["toxicity"])
    
    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        toxicity_results = check_toxicity(prompt)
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "toxicity": toxicity_results
        })
        
        with st.chat_message("user"):
            st.write(prompt)
            with st.expander("View toxicity analysis"):
                display_toxicity_metrics(toxicity_results)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": m["role"], "content": m["content"]} 
                        for m in st.session_state.messages
                    ]
                )
                response_content = response.choices[0].message.content
                
                ai_toxicity = check_toxicity(response_content)
                
                st.write(response_content)
                with st.expander("View toxicity analysis"):
                    display_toxicity_metrics(ai_toxicity)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "toxicity": ai_toxicity
                })

def api_testing_tab():
    st.header("🔍 Toxicity Classification API Testing")
    
    st.subheader("Single Text Analysis")
    text_input = st.text_area("Enter text to analyze:", height=100)
    
    if st.button("Analyze"):
        if text_input:
            with st.spinner("Analyzing..."):
                results = check_toxicity(text_input)
                display_toxicity_metrics(results)
        else:
            st.warning("Please enter some text to analyze.")
    
    st.subheader("Batch Analysis")
    uploaded_file = st.file_uploader("Upload a text file (one text per line)", type=["txt"])
    
    if uploaded_file and st.button("Analyze Batch"):
        with st.spinner("Analyzing batch..."):
            texts = [line.decode("utf-8").strip() for line in uploaded_file.readlines()]
            
            try:
                response = requests.post(
                    "http://localhost:8000/predict/batch",
                    json={"texts": texts}
                )
                results = response.json()
                
                df_data = []
                for text, result in zip(texts, results):
                    row = {"Text": text}
                    row.update(result)
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                st.dataframe(df)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="toxicity_analysis.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing batch: {str(e)}")

def main():
    st.title("🤖 Toxic Comment Classifier & ChatBot")
    
    tab1, tab2 = st.tabs(["ChatBot", "API Testing"])
    
    with tab1:
        chatbot_tab()
    
    with tab2:
        api_testing_tab()

if __name__ == "__main__":
    main()
