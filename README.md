# Question_Answering_LLM_Based_on_Sacred_Texts

## Project Overview

The project consisted of creating a Large Language Model capable of acting as a chatbot specializing in question answering related to several holy texts. In particular, it has been trained on the English Standard Version of the Holy Bible from a Github, as well as Swami Sivananda’s translation of the Bhagwad Gita from the Github Repo of Kushal Shah. The chatbot works by using zero-shot classification to classify a question based on preset labels which are mapped to their appropriate responses. This chatbot is also responsible for diverting flow of control to the DistilBERT and BERT models trained with the appropriate religious text. The former produces the direct answer to the user’s question and is combined with the most appropriate verse found by the latter. After this, sentiment analysis is done on user response to gauge whether control stays in the religious model or returns to the general chatbot.

## Usage

1. Clone the repository:
   ```bash
   https://github.com/mbadalbadalian/Deep_Learning_Algorithms.git
