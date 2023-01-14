import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
import openai
import requests
import nltk
from bs4 import BeautifulSoup
from utils import *
import asyncio

# TODO: add option to upload book and author bio
# TODO: add option to select book from gutenberg
# TODO: add option to select author from wikipedia
# TODO: allow user to input his api key

# --------- SESSION STATE ---------

if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0
# --------- Streamlit ---------

st.title("Chat to Books")
st.subheader("Have a conversation with the author of your favorite book")

st.write("This web app uses AI to allow you to query a book in a conversational manner and" \
      + "the AI will reply in the style of the author. You can either upload a txt file with the" \
      + "book's content, the author's bio, or both, or select from a demo option." \
      + "Note there is a limit of 1000 tokens per session." \
      + "To avoid hitting the limit, you can provide your own openai api key :).")

st.warning("This app is still in development. If you have any suggestions or find any bugs, please let me know!")
st.info("Note that a personal API key is required to process new authors or books. If you don't have one, you can use the demo options.")

# Sidebar
st.sidebar.title("User Inputs")

# Dropdown menu
options = ["El Principito - Antoine de Saint-Exupéry", "The Sun Also rises - Ernest Hemingway"]
selected_book = st.sidebar.selectbox("Select a book", options)

if selected_book:
    with st.spinner("Loading book..."):
        demo_data = pd.read_csv("code/train_data.csv")

sample_questions = ["What inspired you to write this book?",
                    "Can you tell me more about the main character?",
                    "What is the main theme of the book?",
                    "Can you tell me about the writing process for this book?",
                    "What message do you hope readers take away from this book?"]

st.sidebar.subheader(f"Here are some sample questions you can try: ")
sample_question = st.sidebar.selectbox("Select a sample question or input your own next", 
                                       sample_questions)

if sample_question:
    prompt = st.sidebar.text_area("Enter your query:", sample_question)
else:
    prompt = st.sidebar.text_area("Enter your query:")

user_api_key = st.sidebar.text_input("Enter your openai api key (optional):")


# Button
if st.button("Ask the author"):
    if selected_book == "El Principito - Antoine de Saint-Exupéry":
        author_name = demo_data["Author_Name"][1]
        author_bio = demo_data["Author_Bio"][1]
        book_name = demo_data["Book_Name"][1]
        book_id = demo_data["Book_ID"][1]
        book_blocks = demo_data["Book_Blocks"][1]
        book_total_word_count = sum(map(int, demo_data['Word_Count'][1].replace('[', '').replace(']', '').split(', ')))
        book_block_token_count = demo_data['Token_Count'][1]
        
        # summarize the book
        with st.spinner("Give me a minute to read the book... (It may take a few minutes for new books)"):
            book_summary = demo_data["Book_Summary"][1]
        
        # Generate the response
        with st.spinner("Generating the response..."):
            completion = chat_with_author(author_bio,
                                          book_summary,
                                          prompt)
            
            response = completion[0]
            tokens_used = completion[1]
            
            st.session_state.total_tokens_used += tokens_used
        
        
    elif selected_book == "The Sun Also rises - Ernest Hemingway":
        author_name = demo_data["Author_Name"][0]
        author_bio = demo_data["Author_Bio"][0]
        book_name = demo_data["Book_Name"][0]
        book_id = demo_data["Book_ID"][0]
        book_blocks = demo_data["Book_Blocks"][0]
        book_total_word_count = sum(map(int, demo_data['Word_Count'][0].replace('[', '').replace(']', '').split(', ')))
        book_block_token_count = demo_data['Token_Count'][0]

        # summarize the book
        with st.spinner("Give me a minute to read the book... (It may take a few minutes for long books)"):
            book_summary = demo_data["Book_Summary"][0]

        # Generate the response
        with st.spinner("Generating the response..."):
            completion = chat_with_author(author_bio,
                                          book_summary,
                                          prompt)

            response = completion[0]
            tokens_used = completion[1]
            
            st.session_state.total_tokens_used += tokens_used

    st.write(response)
    st.write("Session Statistics:")
    st.write(f"Tokens used: {st.session_state.total_tokens_used}")
    
    if st.session_state.total_tokens_used > 1000:
        st.write("You've used up all your tokens for the day. Come back tomorrow or provide your personal openai api key!")
        st.stop()