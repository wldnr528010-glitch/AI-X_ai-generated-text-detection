# Notebooks

# AI Generated Text Detection

## Title
AI가 쓴 글은 장르가 달라도 구분될까?  
Essay와 News 데이터 기반 AI 텍스트 탐지 모델 비교

## 1. Introduction
생성형 AI가 작성한 글이 증가하면서 AI-generated text와 human-written text를 구분하는 문제가 중요해지고 있다. 본 프로젝트는 Essay와 News 두 장르의 데이터를 활용하여 AI 생성 텍스트 탐지 모델을 구현하고, 장르별로 어떤 머신러닝 알고리즘이 더 적합한지 비교하는 것을 목표로 한다.

## 2. Dataset
본 프로젝트에서는 Essay와 News 데이터를 각각 독립적으로 사용하였다. 두 데이터를 하나로 합치지 않고, 장르별로 별도의 모델을 학습시켰다.

| Dataset | Human | AI | Total |
|---|---:|---:|---:|
| Essay | 2,500 | 2,500 | 5,000 |
| News | 2,500 | 2,500 | 5,000 |

Label은 다음과 같이 정의하였다.

| Label | Meaning |
|---|---|
| 0 | Human-written text |
| 1 | AI-generated text |

## 3. Project Design
Essay와 News는 서로 다른 문체적 특성을 가진다. Essay는 개인 경험, 주장, 감정 표현이 많이 나타나는 반면, News는 객관적이고 형식적인 문장 구조를 갖는 경우가 많다.

따라서 두 데이터를 합쳐 하나의 모델을 학습할 경우, 모델이 AI 여부가 아니라 장르 차이를 학습할 가능성이 있다. 이를 방지하기 위해 본 프로젝트에서는 Essay AI Detection Model과 News AI Detection Model을 별도로 설계하였다.

## 4. Methodology
각 데이터셋에 대해 동일한 분석 절차를 적용하였다.

1. 텍스트 전처리
2. Train/Test Split
3. TF-IDF Vectorization
4. 머신러닝 모델 학습
5. Accuracy, Precision, Recall, F1-score, Training Time 비교
6. Confusion Matrix 분석

사용한 모델은 다음과 같다.

- Naive Bayes
- Logistic Regression
- Random Forest
- XGBoost
- Linear SVM

## 5. Results
Essay 데이터와 News 데이터에 대해 각각 모델을 학습한 뒤 성능을 비교하였다.

| Genre | Best Model | Accuracy | F1-score |
|---|---|---:|---:|
| Essay | 추후 입력 | 추후 입력 | 추후 입력 |
| News | 추후 입력 | 추후 입력 | 추후 입력 |

## 6. Conclusion
본 프로젝트는 AI-generated text 탐지 모델이 Essay와 News라는 서로 다른 장르에서도 안정적으로 작동하는지 확인하고자 하였다. 실험 결과를 통해 장르별로 적합한 알고리즘이 달라질 수 있으며, 단순 정확도뿐만 아니라 F1-score와 오분류 사례를 함께 고려해야 함을 확인하였다.
