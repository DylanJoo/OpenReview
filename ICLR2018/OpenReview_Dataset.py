#!/usr/bin/env python
# coding: utf-8

# In[2]:


import openreview
import pandas as pd
import numpy as np
from scratch_util import *


# # Scratch the ICLR 2018 accepted paper list(oral, poster)

# In[5]:


# Scratch the accepted papers' title from ICLR's webpage.
import re
import urllib.request
from bs4 import BeautifulSoup  # Currently, python's standard lib is sufficient for this task.

def extract_oral():
    html = urllib.request.urlopen(
        'https://iclr.cc/Conferences/2018/Schedule?type=Oral'
        ).read().decode('utf-8')
    res = re.findall("<div class=\x22maincardBody\x22>(.+?)</div>", html)
    return res

def extract_poster():
    html = urllib.request.urlopen(
        'https://iclr.cc/Conferences/2018/Schedule?type=Poster'
        ).read().decode('utf-8')
    res = re.findall("<div class=\x22maincardBody\x22>(.+?)</div>", html)
    return res

oral = extract_oral()
poster = extract_poster()
accept = oral+poster


# In[6]:


#Log into openreview.net, and list all the invitations.
client = openreview.Client(baseurl='https://openreview.net', username = "dylanjootw@gmail.com", password="cfdacfda")
invi = openreview.tools.get_submission_invitations(client)
iclr_invi = [item for item in invi if ('ICLR' in item)]  # invitations with ICLR


# In[7]:


BS = "ICLR.cc/2018/Conference/-/Blind_Submission"  #With all the submitted papers.
OR = "ICLR.cc/2018/Conference/-/Paper.*/Official_Review"
PC = "ICLR.cc/2018/Conference/-/Paper.*/Public_Comment"


# # Preparing all the paper submitted to ICLR 2018 
# 
# @Using Pandas DataFrame.

# In[8]:


# Retrieve the paper with oral accpeted.
pid = []
title = []
abstract = []
key = []

for note in openreview.tools.iterget_notes(client, invitation = BS):
    if note.content['title'] in oral:
        pid.append(note.id)
        title.append(note.content['title'])
        abstract.append(note.content['abstract'])
        key.append(note.content['keywords'])

oral_df = pd.DataFrame(data = {'PID': pid, 'Title': title, 'Abstract': abstract, 'Keyword': key})


# In[9]:


# Retrieve the paper with poster accpeted.
pid = []
title = []
abstract = []
key = []

for note in openreview.tools.iterget_notes(client, invitation = BS):
    if note.content['title'] in poster:
        pid.append(note.id)
        title.append(note.content['title'])
        abstract.append(note.content['abstract'])
        key.append(note.content['keywords'])

poster_df = pd.DataFrame(data = {'PID': pid, 'Title': title, 'Abstract': abstract, 'Keyword': key})


# In[10]:


# Retrieve the rejected paper.
pid = []
title = []
abstract = []
key = []

for note in openreview.tools.iterget_notes(client, invitation = BS):
    if note.content['title'] not in accept:
        pid.append(note.id)
        title.append(note.content['title'])
        abstract.append(note.content['abstract'])
        key.append(note.content['keywords'])

reject_df = pd.DataFrame(data = {'PID': pid, 'Title': title, 'Abstract': abstract, 'Keyword': key})


# In[89]:


#Export into CSV file(Opitional)
oral_df.to_csv('ICLR18_oral.csv')
poster_df.to_csv('ICLR18_poster.csv')
reject_df.to_csv('ICLR18_reject.csv')


# # Preparing official reviews in ICLR 2018 

# In[16]:


# Retrieve the reviews from openreview.net.
pid = []
conf = []
rating = []
review = []
title = []

for note in openreview.tools.iterget_notes(client, invitation = OR):
        title.append(note.content['title'])
        review.append(note.content['review'])
        rating.append(note.content['rating'][0])
        # Extract the scores only.
        conf.append(note.content['confidence'][0])
        pid.append(note.forum)

or_df = pd.DataFrame(data = {'PID': pid, 'Title': title, 'Review': review, 'Rating': rating, 'Conf': conf})


# In[18]:


#Export into CSV file(Opitional)
or_df.to_csv('ICLR18_all_reviews.csv')

