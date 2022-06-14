from urllib import request
import streamlit as st
from youtube_comment_scraper_python import *
import pandas as pd
from streamlit_option_menu import option_menu
import time
import math
from textblob import TextBlob
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import numpy as np
import requests
import json
import googleapiclient.discovery
import requests
import os
from cleantext import clean
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
st.set_page_config(page_title = 'Comment Analysis', 
    layout='wide',
    page_icon='<a target="_blank" href="https://icons8.com/icon/84909/youtube">YouTube</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>')
st.markdown(f"<h1 style='text-align: center; color: Tomato;'>YouTube Comment Analysis</h1>", unsafe_allow_html=True)
video_id = st.text_input("Enter YouTube URL")
index = video_id.rfind("=")
id = video_id[index+1:]
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDHuMMpAmSYWe_uYwv0ra2S7FgA94Zhmd0"
youtube = googleapiclient.discovery.build(
api_service_name, api_version, developerKey = DEVELOPER_KEY)
request = youtube.commentThreads().list(
    part="id,snippet",
    maxResults=100,
    order="relevance",
    videoId= id
)
response = request.execute()
authorname = []
comments = []
high_positive=[]
high_negative=[]
positive = []
negative = []
neutral = []
for i in range(len(response["items"])):
    authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
    comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
    df_1 = pd.DataFrame(comments, index = authorname,columns=["Comments"])
for i in range(len(df_1)):
    text =  TextBlob(df_1.iloc[i,0])
    subjectivity = text.sentiment.subjectivity
    polarity = text.sentiment.polarity
    if subjectivity==0 or polarity==0:
        neutral.append(df_1.iloc[i,0]) 
    elif polarity>0.70:
        high_positive.append(df_1.iloc[i,0])
    elif polarity<-0.70:
        high_negative.append(df_1.iloc[i,0])
    elif polarity<=0.70 and polarity>0:
        positive.append(df_1.iloc[i,0])
    elif polarity>=-0.70 and polarity<0:
        negative.append(df_1.iloc[i,0])
    else:
        neutral.append(df_1.iloc[i,0])
def summarize(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    from sumy.summarizers.text_rank import TextRankSummarizer
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 2)
    text_summary = ""
    for sentence in summary:
        text_summary += str(sentence)
    return text_summary
def rankFinder(comment):
    list_1 = []
    for ele in comment:
        t=TextBlob(ele)
        list_1.append(t.sentiment.polarity)
        if len(list_1)>0:
            return abs(sum(list_1)/len(list_1))
        else:
            pass
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
with st.sidebar:
    selected = option_menu("", ['Dashboard','video', 'Classification','Summarization','Analysis'], 
        icons=['file-earmark-bar-graph','youtube','hr','layout-text-window-reverse','star-fill'], menu_icon="cast", default_index=0)
if selected=="Dashboard":
    st.markdown("## Comments Percentage")
    count=len(positive)+len(negative)+len(neutral)+len(high_positive)+len(high_negative)
    rank_positive=positive+high_positive
    rank_negative=negative+high_negative
    crank=len(rank_negative)+len(rank_positive)+len(neutral)
    cout=len(rank_positive)/(len(rank_positive)+len(rank_negative))
    cin=len(rank_negative)/(len(rank_positive)+len(rank_negative))
    first_kpi, second_kpi, third_kpi,fourth_kpi,fifth_kpi = st.columns(5)
    with first_kpi:
        st.markdown("**Highly Positive**")
        number1 = round(len(high_positive)/count,2)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
    with second_kpi:
        st.markdown("**Positive**")
        number2 = round(len(positive)/count,2)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)
    with third_kpi:
        st.markdown("**Neutral**")
        number3 = round(len(neutral)/count,2)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)
    with fourth_kpi:
        st.markdown("**Negative**")
        number4 = round(len(negative)/count,2)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)
    with fifth_kpi:
        st.markdown("**Highly Negative**")
        number5 = round(len(high_negative)/count,2)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{number5}</h1>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## Comment Graph")
    labels = 'Positive', 'Negative', 'Neutral'
    number6=round(len(rank_positive)/crank,2)
    number7=round(len(rank_negative)/crank,2)
    number8=round(len(neutral)/crank,2)
    sizes = [number6,number7,number8]
    explode = (0,0,0)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
    ax1.axis('equal') 
    st.pyplot(fig1)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## Analysis")
    rank_positive=positive+high_positive
    rank_negative=negative+high_negative
    if len(rank_negative)==0:
        rank_positive=positive+high_positive
        rank_negative=negative+high_negative
        cout2=len(rank_positive)/(len(rank_positive)+len(rank_negative))
        cin=len(rank_negative)/(len(rank_positive)+len(rank_negative))
        rank_1=rankFinder(rank_positive)
        temp_1=rank_1*cout2
        po_int= 1/(1 + np.exp(-temp_1))
        st.markdown(f"<p style='color: black;'>Out of {len(cout2)} % classified positive comments the viewers emotional intensity is : {po_int}'</p>", unsafe_allow_html=True)
    if len(rank_negative)>0:
        rank_positive=positive+high_positive
        rank_negative=negative+high_negative
        cout1=len(rank_positive)/(len(rank_positive)+len(rank_negative))
        cin1=len(rank_negative)/(len(rank_positive)+len(rank_negative))
        rank_1=rankFinder(rank_positive)
        temp_2=rank_1*cout1
        temp_3=rankFinder(rank_negative)
        temp_4=temp_3*cin1
        p_int=temp_2/(temp_2+temp_4)
        n_int=temp_4/(temp_2+temp_4)
        st.markdown(f"<p style='color: black;'>Out of {round(cout1,2)} % positive comments the viewers emotional intensity is : {p_int} .</p>",unsafe_allow_html=True)
        st.markdown(f"<p style='color: black;'>Out of {round(cin1,2)} % negative comments the viewers emotional intensity is : {n_int} .</p>",unsafe_allow_html=True)
if selected == "video":
    st.video(video_id)
if selected=="Classification":     
    selected = option_menu("",["5⭐","4⭐",'3⭐', '2⭐','1⭐'], 
        icons=['', '','',''],default_index=0, orientation="horizontal",
        styles={
        "container": {"padding": "0!important", "background-color": "white"},
        "icon": {"color": "DarkMagenta", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "Tomato"},
    })
    if selected == "5⭐":
        for x in high_positive:
            st.markdown(f"<p style='color: black;'>{x}</p>", unsafe_allow_html=True)
            st.markdown(f"<hr/>",unsafe_allow_html=True)
    if selected == "4⭐":
        for x in positive:
            st.markdown(f"<p style='color: black;'>{x}</p>", unsafe_allow_html=True)
            st.markdown(f"<hr/>",unsafe_allow_html=True)
    if selected == "3⭐":
        for x in neutral:
            st.markdown(f"<p style='color: black;'>{x}</p>", unsafe_allow_html=True)
            st.markdown(f"<hr/>",unsafe_allow_html=True)
    if selected=="2⭐":
        for x in negative:
            st.markdown(f"<p style='color: black;'>{x}</p>", unsafe_allow_html=True)
            st.markdown(f"<hr/>",unsafe_allow_html=True) 
    if selected=="1⭐":
        for x in high_negative:
            st.markdown(f"<p style='color: black;'>{x}</p>", unsafe_allow_html=True)
            st.markdown(f"<hr/>",unsafe_allow_html=True)     
if selected=="Summarization":
    selected = option_menu("",["Positive_Summary",'Negative_Summary', 'Neutral_Summary'], 
    icons=['emoji-laughing', 'emoji-angry','emoji-expressionless'], default_index=0, orientation="horizontal",
    styles={
    "container": {"padding": "0!important", "background-color": "white"},
    "icon": {"color": "Navy", "font-size": "25px"}, 
    "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "Tomato"},
    })
    if selected == "Positive_Summary":
        rank_positive=positive+high_positive
        out=[]
        positive_summary = summarize(rank_positive)
        for ele in positive_summary.split(","):
            out.append(clean(ele,no_emoji=True))
        s = " "
        ans= s.join(out)
        punc = '''!()-[]{};:'"\<>/?@#$%^&*_~'''
        for ele in ans:
            if ele in punc:
                ans = ans.replace(ele, "")
        st.write(ans)
    if selected == "Negative_Summary":
        rank_negative=negative+high_negative
        out=[]
        negative_summary = summarize(rank_negative)
        for ele in negative_summary.split(","):
            out.append(clean(ele,no_emoji=True))
        s = " "
        ans= s.join(out)
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ele in ans:
            if ele in punc:
                ans = ans.replace(ele, "")
        st.write(ans)
    if selected == "Neutral_Summary":
        out=[]
        neutral_summary = summarize(neutral)
        for ele in neutral_summary.split(","):
            out.append(clean(ele,no_emoji=True))
        s = " "
        ans= s.join(out)
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ele in ans:
            if ele in punc:
                ans = ans.replace(ele, "")
        st.write(ans)
if selected=="Analysis":
        if len(negative)==0:
            rank_positive=positive+high_positive
            rank_negative=negative+high_negative
            cout2=len(rank_positive)/(len(rank_positive)+len(rank_negative))
            cin=len(rank_negative)/(len(rank_positive)+len(rank_negative))
            rank_1=rankFinder(rank_positive)
            temp_1=rank_1*cout2
            pos_int= 1/(1 + np.exp(-temp_1))
            if pos_int>=0.98:
                lottie_hello = load_lottieurl('https://assets6.lottiefiles.com/datafiles/QDHTh1tUmPJvYoz/data.json')
                st_lottie(lottie_hello, key="hello")
            if pos_int>0.95 and pos_int<0.98:
                lottie_hello = load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_rwmxkmtj.json')
                st_lottie(lottie_hello, key="hello1")
            if pos_int>0.85 and pos_int<0.95:
                lottie_hello = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_zihz0qdx.json')
                st_lottie(lottie_hello, key="hello2")
            if pos_int>0.50 and pos_int<0.85:
                lottie_hello = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_it5p1n0q.json')
                st_lottie(lottie_hello, key="hello3")
            if pos_int<0.50:
                lottie_hello = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_9sa32yjl.json')
                st_lottie(lottie_hello, key="hello4")
        if len(negative)>0:
            rank_positive=positive+high_positive
            rank_negative=negative+high_negative
            cout1=len(rank_positive)/(len(rank_positive)+len(rank_negative))
            cin1=len(rank_negative)/(len(rank_positive)+len(rank_negative))
            rank_1=rankFinder(rank_positive)
            temp_2=rank_1*cout1
            temp_3=rankFinder(rank_negative)
            temp_4=temp_3*cin1
            p_int=temp_2/(temp_2+temp_4)
            n_int=temp_4/(temp_2+temp_4)
            if p_int>=0.90:
                lottie_hello = load_lottieurl('https://assets6.lottiefiles.com/datafiles/QDHTh1tUmPJvYoz/data.json')
                st_lottie(lottie_hello, key="hello")
            if p_int>=0.80 and p_int<0.90:
                lottie_hello = load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_rwmxkmtj.json')
                st_lottie(lottie_hello, key="hello1")
            if p_int>=0.50 and p_int<0.80:
                lottie_hello = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_zihz0qdx.json')
                st_lottie(lottie_hello, key="hello2")
            if p_int>=0.30 and p_int<0.50:
                lottie_hello = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_it5p1n0q.json')
                st_lottie(lottie_hello, key="hello3")
            if p_int<0.30:
                lottie_hello = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_9sa32yjl.json')
                st_lottie(lottie_hello, key="hello4")