from urllib import request
import streamlit as st
from youtube_comment_scraper_python import *
import pandas as pd
from PIL import Image
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
try:
    st.set_page_config(page_title = 'Comment Analysis', 
        layout='wide',
        page_icon='<a target="_blank" href="https://icons8.com/icon/84909/youtube">YouTube</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>')
    image = Image.open('logo.jpg')
    st.image(image,width=50)
    time.sleep(5)
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
    positive = []
    negative = []
    objective=[]
    neutral = []
    for i in range(len(response["items"])):
        authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
        comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
        df_1 = pd.DataFrame(comments, index = authorname,columns=["Comments"])
    for i in range(len(df_1)):
        text =  TextBlob(df_1.iloc[i,0])
        subjectivity = text.sentiment.subjectivity
        polarity = text.sentiment.polarity
        if subjectivity > 0.4:
            if polarity > 0.1 :
                positive.append(df_1.iloc[i,0]) 
            elif polarity < -0.1:
                negative.append(df_1.iloc[i,0])  
            else:
                neutral.append(df_1.iloc[i,0])
        elif subjectivity < 0.4 and polarity > 0.2 :
            positive.append(df_1.iloc[i,0])
        else:
            objective.append(df_1.iloc[i,0])
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
    rank_1 = rankFinder(positive)
    with st.sidebar:
        selected = option_menu("", ['Dashboard','video', 'Classification','Summarization','Analysis'], 
            icons=['file-earmark-bar-graph','youtube','hr','layout-text-window-reverse','star-fill'], menu_icon="cast", default_index=0)
    if selected=="Dashboard":
        st.markdown("## Comments Percentage")
        count=len(positive)+len(negative)+len(neutral)+len(objective)
        first_kpi, second_kpi, third_kpi,fourth_kpi = st.columns(4)
        with first_kpi:
            st.markdown("**Positive**")
            number1 = round(len(positive)/count,2)
            st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
        with second_kpi:
            st.markdown("**Negative**")
            number2 = round(len(negative)/count,2)
            st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)
        with third_kpi:
            st.markdown("**Neutral**")
            number3 = round(len(neutral)/count,2)
            st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)
        with fourth_kpi:
            st.markdown("**Objective**")
            number4 = round(len(objective)/count,2)
            st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("## Comment Graph")
        labels = 'Positive', 'Negative', 'Neutral','objective'
        sizes = [number1,number2,number3,number4]
        explode = (0,0,0,0)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax1.axis('equal') 
        st.pyplot(fig1)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("## Analysis")
        if len(negative)==0:
            temp_1=rank_1*len(positive)
            po_int= 1/(1 + np.exp(-temp_1))
            st.markdown(f"<p style='color: black;'>The Number of Positive Comments are : {len(positive)} and the Intensity of Positive Comments is : {po_int}</p>", unsafe_allow_html=True)
        if len(negative)>0:
            temp_2=rank_1*len(positive)
            temp_3=rankFinder(negative)
            temp_4=temp_3*len(negative)
            p_int=temp_2/(temp_2+temp_4)
            n_int=temp_4/(temp_2+temp_4)
            st.markdown(f"<p style='color: black;'>The number of positive comments are : {len(positive)} and the intensity of positive comments is : {p_int} .</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: black;'>The number of negative comments are : {len(negative)} and the intensity of negative comments is : {n_int} .</p>",unsafe_allow_html=True)
    if selected == "video":
        st.video(video_id)
    if selected=="Classification":     
        selected = option_menu("",["Positive",'Negative', 'Neutral','objective'], 
            icons=['plus-lg', 'dash','plus-slash-minus','boxes'], default_index=0, orientation="horizontal",
            styles={
            "container": {"padding": "0!important", "background-color": "white"},
            "icon": {"color": "DarkMagenta", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "HotPink"},
        })
        if selected == "Positive":
            for x in positive:
                st.markdown(f"<p style='color: black;'>{x}</p>", unsafe_allow_html=True)
                st.markdown(f"<hr/>",unsafe_allow_html=True)
        if selected == "Negative":
            for x in negative:
                st.markdown(f"<p style='color: black;'>{x}</p>", unsafe_allow_html=True)
                st.markdown(f"<hr/>",unsafe_allow_html=True)
        if selected == "Neutral":
            for x in neutral:
                st.markdown(f"<p style='color: black;'>{x}</p>", unsafe_allow_html=True)
                st.markdown(f"<hr/>",unsafe_allow_html=True)
        if selected=="objective":
            for x in objective:
                st.markdown(f"<p style='color: black;'>{x}</p>", unsafe_allow_html=True)
                st.markdown(f"<hr/>",unsafe_allow_html=True)       
    if selected=="Summarization":
        selected = option_menu("",["Positive_Summary",'Negative_Summary', 'Neutral_Summary'], 
        icons=['emoji-laughing', 'emoji-angry','emoji-expressionless'], default_index=0, orientation="horizontal",
        styles={
        "container": {"padding": "0!important", "background-color": "white"},
        "icon": {"color": "Navy", "font-size": "25px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "Aquamarine"},
        })
        if selected == "Positive_Summary":
            out=[]
            positive_summary = summarize(positive)
            for ele in positive_summary.split(","):
                out.append(clean(ele,no_emoji=True))
            s = " "
            ans= s.join(out)
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for ele in ans:
                if ele in punc:
                    ans = ans.replace(ele, "")
            st.write(ans)
        if selected == "Negative_Summary":
            out=[]
            positive_summary = summarize(negative)
            for ele in positive_summary.split(","):
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
            positive_summary = summarize(neutral)
            for ele in positive_summary.split(","):
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
                rank_1 = rankFinder(positive)
                inten_1=rank_1*len(positive)
                neg_int=0
                pos_int= 1/(1 + np.exp(-inten_1))
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
                rank_1 = rankFinder(positive)
                inten_1=rank_1*len(positive)
                rank_2=rankFinder(negative)
                inten_2=rank_2*len(negative)
                pos_int=inten_1/(inten_1+inten_2)
                neg_int=inten_2/(inten_1+inten_2)
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
except:
    pass
        
