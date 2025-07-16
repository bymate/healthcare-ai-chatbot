# 🤖 RAG + PEFT 활용 헬스케어 LLM 챗봇 구현
<br/>

## 🧑‍💻 프로젝트 개요  
해당 프로젝트는 간단하고 반복적인 질문에 대하여 의료 서비스의 접근성과 진료 자원의 효율성을 높이기 위한 관점으로 시작했습니다. 헬스케어 질의응답 데이터를 기반으로, 사용자 질문에 대해 정확한 답변을 제공하는 AI 헬스케어 챗봇을 구축하는 것을 목표로 진행했습니다. **키워드 중요도와 질문-답변 유사도**를 활용해 **질문-답변 쌍을 구성**하는 매칭 알고리즘을 설계해 **학습 데이터의 품질**을 높이고, **RAG 방식**을 통해 **LLM 모델의 답변 성능**을 보완했습니다.
<br/><br/><br/>

## 📦 데이터셋  
**데이터 수집 및 구성**  
- 초거대 AI 헬스케어 질의응답 데이터(AI-Hub)
- 답변은 **answer, body, conclusion**으로 구성
- 의도 외에는 **어떤 질문에 대한 답변인지 알 수 없음**
<img width="1062" height="330" alt="image" src="https://github.com/user-attachments/assets/a7c9d19a-50e0-42e9-a1e3-d07b678beb12" />

**질문-답변 쌍 데이터셋 생성**

<img width="828" height="900" alt="image" src="https://github.com/user-attachments/assets/4797e892-e440-4b9b-bbd0-c479f8d61125" />
<br/><br/><br/>

## 🛠️ 모델 학습 상세 설명  
**학습 모델**
- 질문 의도 분류: `RandomForestClassifier`
- LLM: `beomi/Llama-3-Open-Ko-8B-Instruct-preview`

**모델 학습**
- LLM 모델 양자화, LoRA 기법을 활용해 fine tuning
- 분류 모델을 활용한 질문 의도 분류

**응답 구조**
- 질문 의도 분류 모델을 통해 질문 의도 파악
- 답변 벡터 DB에서 해당 **의도에 해당하는 답변 벡터 DB 재생성**
- **RAG 방식**을 활용해 질문과 유사도가 높은 답변 3개 추출  
- Llama3기반의 LLM 모델이 **검색된 정보를 활용해 응답 생성**
<br/>

## 🧩 주요 역할 및 프로젝트 진행 내용  
- 헬스케어 질의응답 학습 데이터 구성(데이터 전처리) 
- 키워드 및 문장 유사도 기반 질문-답변 매칭 로직 구현
- PEFT(LoRA) 기법으로 Llama3 모델 fine tuning 메모리 사용량 및 학습 시간 단축
- 질문 의도 분류 모델 학습
- RAG 시스템 구성 및 응답 구조 설계  
- 실제 사용자 입력에 대해 챗봇 응답 성능 평가
<br/>

## ⚙️ 사용 기술 스택  
- Python
- PEFT(LoRA)
- RAG (Retrieval-Augmented Generation)  
- Llama3
- scikit-learn, RandomForestClassifier, pandas
<br/>

## 🎯 Result
<img width="793" height="619" alt="image" src="https://github.com/user-attachments/assets/a68935bd-14e3-4fc9-8cb1-8de2074f00a1" />
<br/><br/><br/>

**[모델 성능 평가]**

**AI 챗봇 성능 평가 방식**
- 평가용 레퍼런스를 기준으로 데이터 제공 회사의 LLM 모델과 프로젝트 모델 비교

**평가용 레퍼런스**
- 학습에 사용한 데이터 셋이 두 모델이 다르므로 평가용 레퍼런스를 생성해 평가 진행
- RAG 기반 참고 답변 5개 추출
- GPT Open API (gpt-3.5-turbo)로 답변 생성(질문 + 참고 정보)

**AI 챗봇 성능 평가 지표**
- Bert Score, Bleu, Rouge

**평가 결과**
- 제공 모델에 비해 **Bleu 약 1.8배, RougeL 1.3배 개선**
- 레퍼런스가 다른 모델을 사용해 만든 답변이기에 절대적인 점수는 낮으나, 동일한 조건에서 상대적인 개선 효과는 확인이 됨.
<img width="514" height="223" alt="image" src="https://github.com/user-attachments/assets/ca1ae63c-2a03-45de-b357-5fc0bccde297" />
<br/><br/><br/>

[코드 파일 순서]

```
data_preprocess_origin.ipynb

match_questions_answers.ipynb

LLM fine tuning.ipynb

의도분류모델.ipynb

의도분류&RAG.ipynb

makebot.ipynb

최종모델_성능평가.ipynb
