import pandas as pd
import numpy as np
import json
import re
import os
import openai
import requests
import nltk
from bs4 import BeautifulSoup
import asyncio
import more_itertools


def get_author_bio(author_name:str)->str:
    base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    url = base_url + author_name
    response = requests.get(url)
    data = json.loads(response.text)
    extract = data['extract']
    return extract

def save_author_bio(author_name:str, bio:str,metadata_folder:str)->None:
    filename = os.path.join(metadata_folder, author_name + ".txt")
    with open(filename, "w") as f:
        f.write(bio)

def get_and_save_author_bio(author_name:str,metadata_folder:str)->None:
    bio = get_author_bio(author_name)
    save_author_bio(author_name, bio,metadata_folder)

def download_book_by_id(book_id: int, book_folder: str, save_as: str) -> None:
    url = f'http://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
    response = requests.get(url)
    if response.status_code == 404:
        print(f'Book with id {book_id} not found')
        return
    book_name = f'{save_as}-{book_id}.txt'
    filename = os.path.join(book_folder, book_name)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f'Book {book_name} has been downloaded.')
    
def build_dataset(author_name, author_bio, book_name, book_id, book_content) -> pd.DataFrame:
    
    df = pd.DataFrame({"Author_Name": [author_name], 
                       "Author_Bio": [author_bio], 
                       "Book_Name": [book_name], 
                       "Book_ID": [book_id], 
                       "Book_Content": [book_content]})
    
    return df    

def get_book_blocks(book_content, block_size=512):
    
    tokenized_book = nltk.word_tokenize(book_content)
    book_blocks = [" ".join(tokenized_book[i:i+block_size]) for i in range(0, len(tokenized_book), block_size)]
    
    return book_blocks

def clean_book_blocks(book_blocks):
    
    # remove empty blocks
    book_blocks = [block for block in book_blocks if block.strip() != ""]
    
    # remove blocks with less than 20 tokens
    book_blocks = [block for block in book_blocks if len(nltk.word_tokenize(block)) > 20]
    
    # remove special characters such as \n, \t, \r
    book_blocks = [block.replace("\n", " ").replace("\t", " ").replace("\r", " ") for block in book_blocks]
    
    # remove multiple spaces
    book_blocks = [re.sub(' +', ' ', block) for block in book_blocks]
    
    # make lowercase
    book_blocks = [block.lower() for block in book_blocks]
    
    return book_blocks

def block_word_count(book_blocks):
    
    word_count = [len(re.findall(r'\w+', block)) for block in book_blocks]
    
    return word_count

def block_token_count(book_blocks):
    
    token_size = [len(block.split()) for block in book_blocks]
    
    return token_size

async def summarize_block(api_key,
                          block,
                          max_tokens=200,
                          temperature=0.5,
                          top_p=1.0,
                          frequency_penalty=0.0,
                          presence_penalty=0.0):
    
    openai.api_key = api_key
    
    completions = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"Summarize the following chunk of a book and keep the essence of the author. Keep in mind that there are multiple chunks and they are being fed sequentially: {block}",
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
      stop=["\n", " #"]
    )
    
    response = completions.choices[0].text
    tokens_used = len(response.split())
    
    return (response, tokens_used)

async def summarize_book(book_content, 
                   block_size=2000, 
                   max_tokens=200, 
                   temperature=0.5, 
                   top_p=1.0, 
                   frequency_penalty=0.0, 
                   presence_penalty=0.0):
    
    book_blocks = get_book_blocks(book_content)
    cleaned_book_blocks = clean_book_blocks(book_blocks)
    book_summary = ""
    
    # Split the cleaned_book_blocks into batches of 5 blocks each
    batches = more_itertools.chunked(cleaned_book_blocks, 5)
    
    for batch in batches:
        tasks = [asyncio.ensure_future(summarize_block(block)[0]) for block in batch]
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            book_summary += response
    
    return book_summary


def chat_with_author(api_key,
                     author_bio, 
                     book_summary,
                     query,
                     max_tokens=100, 
                     temperature=0.5, 
                     top_p=1.0, 
                     frequency_penalty=0.0, 
                     presence_penalty=0.0):
    
    openai.api_key = api_key
    
    prompt = f"Based on the following author biography {author_bio} and book summary {book_summary}" \
       + f"Adopt the personality of the author and be capable of answering questions about the book. " \
       + f"For example, if the question is 'What is the name of the book?', the answer should be the name of the book. " \
       + f"Another example is 'What is the name of the author?', the answer should be the name of the author. " \
       + f"Another example is 'What is the book about?', the answer should be the summary of the book. " \
       + f"Finally, other examples can be more specific about the book's content. " \
       + "The important thing is to converse as if you are the author of the book. " \
       + "[Question]" + f"{query}" + "[Answer]" + " "
       
    completions = openai.Completion.create(engine="text-davinci-003", 
                                        prompt=prompt, 
                                        max_tokens=100, 
                                        n=1,
                                        stop=None,
                                        temperature=0.5)

    response = completions.choices[0].text
    # count tokens used
    tokens_used = len(response.split())
    
    return (response, tokens_used)