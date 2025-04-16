import streamlit as st
import pandas as pd
import torch
import joblib
import pandas as pd
import sys
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, # 메모리 사용량을 줄이기 위해 양자화
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from SelectQuestion import SelectOneQuestion # 직접 만든 클래스
from sklearn.metrics.pairwise import cosine_similarity

# 학습된 모델 사용을 위한 기본 설정 
# 학습된 모델 및 토크나이저 불러오기
model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/full_trained_model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/full_trained_model")

# 특정 디렉토리를 모듈 검색 경로에 추가
sys.path.append('/content/drive/MyDrive/healthcare project')

# 학습된 분류 모델 불러오기
rf_model = joblib.load('/content/drive/MyDrive/healthcare project/headache_randomforest.pkl')
scaler = joblib.load('/content/drive/MyDrive/healthcare project/headache_scaler.pkl')
pca = joblib.load('/content/drive/MyDrive/healthcare project/headache_pca.pkl')

# 클래스 동작을 위한 정의
question_df = pd.read_csv("/content/drive/MyDrive/healthcare project/두통 질문.csv", index_col=0)
embed_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
labelencoder = LabelEncoder()

# 객체 생성
select_question = SelectOneQuestion(question_df, scaler, pca, rf_model, embed_model, labelencoder)
# 벡터 DB 만들기
intention_map, question_map = select_question.make_vec_db()


# 코사인 유사도 구하는 함수
def krsbert_similarity(question, result_question):

    # 모델 불러오기
    user_question_embedding = embed_model.encode(question) # 사용자 질문 임베딩
    select_question_embedding = embed_model.encode(result_question) # 추출된 질문 임베딩
    cos_sim = cosine_similarity(user_question_embedding.reshape(1,-1), select_question_embedding.reshape(1,-1)) # 코사인 유사도

    return cos_sim[0][0]

# 파이프라인 미리 생성(답변 출력할 때 마다 생성하는 것을 막기 위함.)
def load_pipe(model, tokenizer):
    return pipeline(task='text-generation', model=model, tokenizer=tokenizer, max_length=256)

# 답변 생성 함수
def generate_answer(user_input):

    question = user_input
    result_question = select_question.search_one_question(question, intention_map, question_map)

    # 유사도 점수 0.84 이상의 질문들만
    if krsbert_similarity(question, result_question).round(2) >= 0.84:
        filtered_question = result_question

    else:
        filtered_question = question

    # 답변 생성
    query = filtered_question
    pipe = load_pipe(model, tokenizer)
    result = pipe(f"<s>[INST]{query}[/INST]")
    answer_result = result[0]['generated_text']
    answer = answer_result.split('[/INST]')[1].split('<')[0]

    # 마침표로 안끝나는 경우 마지막 마침표까지만 출력력
    result = answer[:answer.rfind('.')+1]

    return result

    

st.title("🧠 의료 AI 챗봇")

with st.chat_message("assistant"):
    st.write("안녕하세요. 의료 AI 챗봇입니다. 궁금한 부분을 말씀해주세요.👋")

prompt = st.chat_input("메시지를 입력하세요.")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    response = generate_answer(prompt)

    # 🤖 챗봇 메시지 출력
    with st.chat_message("assistant"):
        st.write(response)