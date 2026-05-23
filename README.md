# AI-X_ai-generated-text-detection

## Title: AI 생성 콘텐츠 탐지 모델 비교 분석  
### AI 글·뉴스를 머신러닝으로 구분할 수 있을까?

---

## Members
```
김희연, 2023073081
손승한, 2024020891
우지욱, 2023025969
임준형, 
```
---

## Project Overview

본 프로젝트는 ChatGPT와 같은 생성형 AI가 작성한 텍스트와 사람이 작성한 텍스트를 구분하는 머신러닝 모델을 구현하고, 모델별 성능을 비교하는 것을 목표로 한다.

최근 생성형 AI의 활용이 확대되면서 AI가 작성한 글, 뉴스, 게시글, 과제물 등을 탐지하는 문제가 중요해지고 있다. 이에 따라 본 프로젝트에서는 공개 데이터셋을 활용하여 AI-generated text와 human-written text를 분류하고, 여러 머신러닝 알고리즘의 성능을 비교한다.

---

## Dataset

본 프로젝트에서는 AI 생성 텍스트와 인간 작성 텍스트가 라벨링된 공개 데이터셋을 활용한다.

예상 활용 데이터셋:

- LLM - Detect AI Generated Text
- NYT AI Generated vs Human Text

데이터는 다음과 같은 형태로 구성한다.

| Column | Description |
|---|---|
| text | 분석 대상 텍스트 |
| label | AI 생성 여부 |
| source | 데이터 출처 |
| category | 텍스트 유형 |

라벨은 다음과 같이 정의한다.

| Label | Meaning |
|---|---|
| 0 | Human-written text |
| 1 | AI-generated text |

---

## Methods

본 프로젝트에서는 텍스트 데이터를 전처리한 뒤, TF-IDF 방식으로 벡터화하고 여러 머신러닝 모델을 학습시킨다.

사용 예정 모델:

- Naive Bayes
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine

성능 평가는 다음 지표를 활용한다.

- Accuracy
- Precision
- Recall
- F1-score
- Training Time
- Confusion Matrix

---

## Repository Structure
