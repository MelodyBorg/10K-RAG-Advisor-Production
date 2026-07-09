**Final Project – Build and Evaluate RAG Chatbots**

# Overview

In this assignment, your team will be building a chatbot based on RAG and Large Language Models that compare the three companies, Alphabet(Google), Amazon, and Microsoft, based on their latest 10K files [here](https://drive.google.com/drive/folders/1kvTWJK_NPlXvLrkH9pgIUO5cimAd6rl0?usp=sharing).

A **10-K** is an annual financial report that publicly traded companies in the U.S. must file with the **Securities and Exchange Commission (SEC)**. It provides a comprehensive overview of a company’s financial performance, business operations, risk factors, and management discussion.

You will be provided codes to build a basic chatbot, which you can then improve. Based on the basic chatbot, you can then explore different Language Model (Gemini, Llama, Deepseek, Kimi, GPT, etc.) used in your chatbot, Embedding Model (OllamaEmbeddings, OpenAIEmbeddings, GeminiEmbeddings), and write the system prompt for your model. 

You will be evaluated based on two aspects  
(1) your model’s ability to accurately answer a set of test questions (contributed by your peer groups); and more importantly, with a mindset of making a great product (an assistant chatbot) that is reliable, robust, impressive, and user-friendly. 

(2) sharing your experience with the class on what you learned in this process – such as differences in the performance when using deepseek vs llama models, how you figure out the boundary of the model performance (hallucination). Therefore, even if your model does not perform the best, but you have great insights and story to share, you might still win\!

# Major steps

## Step-by-step guidance is Available on the Github repository here: [https://github.com/JHU-CDHAI/Chatbot-AIEB](https://github.com/JHU-CDHAI/Chatbot-AIEB) 

The repository contains detailed step-by-step instructions for:

* Installing required tools (VSCode, Miniconda)  
* Setting up your Python environment with all necessary python packages  
* Configuring access to local (ollama) and cloud-based LLMs (gemini)  
* Running sample applications to test your setup

By following the instructions in this repository, you will:

* Set up your development environment with VSCode, Miniconda, and necessary Python packages  
* Access different LLM options:  
* Run local LLMs using Ollama (e.g., Mistral, Llama3.1)  
* Connect to cloud-based LLMs via OpenAI and Google Gemini APIs  
* Create interactive chatbot interfaces using Streamlit as the frontend and various LLMs as the backend


## In the repository, you will find the codes for the Basic RAG Chatbot:

[https://github.com/JHU-CDHAI/Chatbot-AIEB/blob/main/chat\_with\_pdf\_gemini\_with\_history.py](https://github.com/JHU-CDHAI/Chatbot-AIEB/blob/main/chat_with_pdf_gemini_with_history.py)

After you can successfully run this one, you can start to customize your own chatbot. 

# Sample Questions

## Risk Evaluation 

Q1: Do these companies worry about the challenges or business risks in China or India in terms of cloud service. 

## Number Checking

Q2: How much CASH does Amazon have at the end of 2025\.

Q3: Compared to 2024, does Amazon's liquidity decrease or increase. 

## Cloud Computing versus other Business

Q4: What is the business where main revenue comes from for Amazon / Google / Microsoft?

Q5: What main businesses does Amazon do?

# Project Deliverables 

You will submit the following: 

Engineering Part: 

- Your codes, and chatbot can work well. 

Business Part: 

Stage1:

- **10k files.** (Understand 10k files first, NotebookLM)  
- Step1: Come up with **a list of questions** that the 10k files have the answers,   
- Step2: then test your chatbot, and **finetune/update** the prompts, persona, or different types of models to make sure your chatbot can pass your questions.  
  - You need to try questions and make sure your chatbot can work well. The answer is correct.   
  - Ask some meaningful questions, and interesting questions.   
  - (Your can turn to NotebookLM to better understand 10k  
- Step3: (One of them should be used to test other teams). 

Stage2:

- Starting with the powerful settings (best LLM or best Embedding). And then compare it with other settings. 

Stage3:

- And also find another question to explore the boundaries of your chatbot, you can fool it, cheat it whatever to explore its boundary. And share some insights. 

Final Deliverables

1. **Your codes**: Either link your GitHub repository or submit the code files. (We prefer the Github link so you can put it in your CV to impress your interviewers) Note: make sure you are careful about adding your API key (like OpenAI key or Gemini key) on a public repo.   
2. **Tech note:**   
   1. A brief explanation of your approach (model choice, system prompt, and architecture).   
      1. RAG: chunksize, embeddings models. Vector\_store.   
   2. Please include any insights you gained (e.g. approaches that failed), and an assessment of your model’s strengths and weaknesses. Include the names of your team members and their roles.  
3. **Presentation**: A 8-minute team presentation summarizing your design choices and results.  
   1. Try different settings  
   2. **Explore the boundaries of your Chatbot, any hallucinations.**  
      1. **How to further reduce hallucinations.**   
   3. What challenges you meet and how you solve them.  
   4. Explore the boundaries of your chatbot: give up a case that the LLM returns the hallucinations.  
4. **Design a Question to Challenge Others:**  that you think other groups are hard to answer, but your ChatBot can answer. On the presentation day, you will rate other groups’ chatbots with your hard questions.

# Tips

* Starting with the most powerful models (like Gemini and OpenAI) to pass your proposed questions.   
    
* Then replace them with less strong models (local models with Ollama) to check the performance differences.   
    
* Ollama embedding models might be slow. You can replace **Ollama embedding models** with **Gemini embedding models**. Gemini embedding models and Ollama LLM models can work together in one RAG Chatbot. 

