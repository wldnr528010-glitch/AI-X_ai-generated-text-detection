# Notebooks

이 폴더는 본 프로젝트의 핵심 분석 코드를 담고 있다.  
프로젝트의 전체 흐름은 `human_vs_ai_text_classification.py` 파일을 중심으로 진행되며, 데이터 전처리부터 시각화, 모델 학습, 최종 성능 비교까지 하나의 코드 안에서 순서대로 실행된다.

본 프로젝트는 생성형 AI가 작성한 텍스트와 사람이 작성한 텍스트를 구분하는 AI-generated text detection 문제를 다룬다. 특히 Essay와 News 데이터를 main experiment로 사용하고, Hard dataset을 additional experiment로 활용하여 데이터의 장르와 난이도에 따라 모델 성능이 어떻게 달라지는지 확인하였다.

---

## 1. Main Code File

```text
human_vs_ai_text_classification.py
```

이 파일은 프로젝트의 메인 실행 코드이다.  
코드는 다음과 같은 흐름으로 구성되어 있다.

```text
1. 라이브러리 불러오기
2. 텍스트 전처리 함수 정의
3. Essay, News, Hard dataset 불러오기
4. 데이터 전처리 및 clean_text 생성
5. Human text와 AI text의 단어 빈도 시각화
6. TF-IDF 기반 고전 머신러닝 모델 학습
7. DistilBERT 기반 모델 학습
8. 전체 모델 성능 비교
9. 결과 이미지 및 CSV 저장
```

단순히 모델을 실행하는 것에서 끝나는 것이 아니라, 각 단계에서 왜 해당 방법을 사용했는지 이해할 수 있도록 주석을 포함하였다.

---

## 2. Why This Code Structure?

본 프로젝트에서는 Essay와 News 데이터를 하나로 합치지 않고 각각 독립적인 분류 문제로 설계하였다.  
그 이유는 Essay와 News가 서로 다른 문체적 특성을 가지기 때문이다.

Essay는 개인적인 경험, 주장, 감정 표현, 연결어 사용이 많이 나타나는 반면, News는 객관적이고 형식적인 문장 구조를 가지는 경우가 많다. 두 데이터를 합쳐 하나의 모델로 학습하면, 모델이 AI 여부가 아니라 장르 차이를 학습할 가능성이 있다.

따라서 본 코드에서는 Essay AI detection model과 News AI detection model을 따로 학습하고, 두 장르에서 모델 성능이 어떻게 달라지는지 비교하였다.

Hard dataset은 main experiment가 아니라 additional experiment로 사용하였다.  
이는 더 어려운 데이터 환경에서 고전 머신러닝 모델과 BERT 기반 모델의 성능 차이를 확인하기 위한 목적이다.

---

## 3. Dataset Path

코드는 프로젝트 최상위 폴더를 기준으로 실행한다고 가정한다.

사용하는 원본 데이터 경로는 다음과 같다.

```text
../data/project_dataset/essay_dataset.csv
../data/project_dataset/news_dataset.csv
../data/project_dataset/hard_dataset.csv
```

각 데이터셋은 다음과 같은 label 구조를 가진다.

| Label | Meaning |
|---|---|
| 0 | Human-written text |
| 1 | AI-generated text |

---

## 4. Preprocessing

텍스트 데이터는 원본 상태 그대로 모델에 입력하기 어렵기 때문에, `clean_text()` 함수를 통해 전처리를 수행하였다.

전처리 과정은 다음과 같다.

```text
1. 깨진 문자 치환
2. 소문자 변환
3. 특수문자 제거
4. 알파벳과 공백만 남기기
5. 불용어 제거
6. Lemmatization
7. 너무 짧은 텍스트 제거
```

이러한 전처리를 적용한 이유는 모델이 불필요한 문자나 표기 차이에 영향을 받기보다, 실제 단어 사용 패턴을 중심으로 학습하도록 만들기 위해서이다.

예를 들어 대문자와 소문자가 섞여 있으면 같은 단어도 서로 다른 단어처럼 처리될 수 있다. 또한 `the`, `is`, `and`와 같은 불용어는 대부분의 문서에서 반복적으로 등장하기 때문에 AI text와 human text를 구분하는 데 큰 도움이 되지 않을 수 있다.

따라서 본 프로젝트에서는 텍스트를 정리한 뒤 `clean_text` 컬럼을 생성하고, 이를 모델 학습에 사용하였다.

---

## 5. Word Frequency Analysis

전처리된 텍스트를 바탕으로 Human-written text와 AI-generated text에서 자주 등장하는 단어를 비교하였다.

각 데이터셋에 대해 다음 시각화를 생성한다.

```text
- Human Top 20 Words
- AI Top 20 Words
- Human WordCloud
- AI WordCloud
```

이 분석은 모델 학습 전에 사람 글과 AI 글 사이에 단어 사용 패턴 차이가 존재하는지 확인하기 위한 과정이다.

생성되는 이미지 파일은 다음과 같다.

```text
../results/figures/essay_words.png
../results/figures/news_words.png
../results/figures/hard_words.png
```

---

## 6. Classical Machine Learning Models

본 프로젝트에서는 먼저 TF-IDF 기반의 고전 머신러닝 모델을 학습하였다.

사용한 모델은 다음과 같다.

```text
1. Logistic Regression
2. Naive Bayes
3. Random Forest
4. XGBoost
```

TF-IDF는 텍스트를 숫자 벡터로 변환하는 방법이다.  
특정 문서에서 자주 등장하지만 전체 문서에서는 흔하지 않은 단어에 높은 가중치를 부여하기 때문에, AI text와 human text의 단어 사용 차이를 반영할 수 있다.

각 모델을 사용한 이유는 다음과 같다.

| Model | Reason |
|---|---|
| Logistic Regression | 텍스트 분류에서 자주 사용되는 기본적인 선형 분류 모델이므로 baseline으로 사용 |
| Naive Bayes | 단어의 등장 확률을 기반으로 분류하는 모델로, 텍스트 분류에서 빠르고 간단하게 적용 가능 |
| Random Forest | 여러 결정 트리를 결합하여 복잡한 패턴을 학습할 수 있는지 확인하기 위해 사용 |
| XGBoost | 강력한 앙상블 모델로, 고전 머신러닝 모델 중 높은 성능을 보일 수 있는지 확인하기 위해 사용 |

고전 모델의 성능 비교 이미지는 다음 경로에 저장된다.

```text
../results/figures/essay_classical.png
../results/figures/news_classical.png
../results/figures/hard_classical.png
```

---

## 7. BERT-based Model

고전 머신러닝 모델과 비교하기 위해 DistilBERT 기반 모델도 사용하였다.

BERT 계열 모델은 단어의 빈도뿐만 아니라 문장 전체의 문맥 정보를 함께 반영할 수 있다.  
따라서 단순한 단어 사용 패턴만으로 구분하기 어려운 데이터에서 더 좋은 성능을 보일 가능성이 있다.

본 프로젝트에서는 실행 시간을 고려하여 DistilBERT를 사용하였다.  
DistilBERT는 BERT의 경량화 모델로, 상대적으로 빠르게 학습할 수 있으면서도 문맥 정보를 반영할 수 있다는 장점이 있다.

코드에는 두 가지 BERT 학습 방식이 포함되어 있다.

| Version | Description |
|---|---|
| Full Version | 더 많은 샘플과 epoch을 사용하지만 실행 시간이 오래 걸림 |
| Light Version | 샘플 수와 epoch을 줄여 빠르게 실행 가능 |

실행 시간이 제한된 경우 Light Version을 사용하는 것을 권장한다.

---

## 8. Evaluation Metrics

모델 성능은 Accuracy와 F1-score를 기준으로 평가하였다.

| Metric | Meaning |
|---|---|
| Accuracy | 전체 데이터 중 모델이 올바르게 예측한 비율 |
| F1-score | Precision과 Recall의 균형을 고려한 성능 지표 |

Accuracy는 전체적인 분류 성능을 확인하기에 직관적인 지표이다.  
그러나 AI text와 human text를 균형 있게 잘 분류하는지도 중요하기 때문에 F1-score도 함께 사용하였다.

---

## 9. Main Experiment and Additional Experiment

본 프로젝트의 실험은 크게 두 가지로 구분된다.

```text
Main Experiment:
- Essay dataset
- News dataset

Additional Experiment:
- Hard dataset
```

Essay와 News는 장르별 AI 탐지 성능을 비교하기 위한 핵심 데이터셋이다.  
Hard dataset은 더 어려운 조건에서 모델 성능이 어떻게 변화하는지 확인하기 위한 추가 실험 데이터셋이다.

이 구분을 통해 프로젝트의 중심 주제인 “장르별 AI text detection”을 유지하면서도, 모델의 난이도별 성능 차이를 함께 살펴볼 수 있다.

---

## 10. Output Files

코드를 실행하면 다음 결과물이 생성된다.

```text
../data/processed/
├── essay_clean_dataset.csv
├── news_clean_dataset.csv
└── hard_clean_dataset.csv

../results/figures/
├── essay_words.png
├── news_words.png
├── hard_words.png
├── essay_classical.png
├── news_classical.png
├── hard_classical.png
└── final_all_models.png

../results/tables/
└── final_results.csv
```

---

## 11. How to Run

먼저 프로젝트 최상위 폴더에서 필요한 라이브러리를 설치한다.

```bash
pip install -r requirements.txt
```

그다음 메인 Python 파일을 실행한다.

```bash
python notebooks/human_vs_ai_text_classification.py
```

코드 실행 후 전처리 데이터, 시각화 이미지, 최종 결과표가 자동으로 생성된다.

---

## 12. Notes

- 일부 데이터 파일은 용량 문제로 GitHub에 업로드되지 않을 수 있다.
- 업로드되지 않은 전처리 데이터는 코드를 실행하면 다시 생성할 수 있다.
- BERT 학습은 실행 환경에 따라 시간이 오래 걸릴 수 있다.
- GPU가 없는 경우 CPU로 실행되며, 학습 시간이 증가할 수 있다.
- 본 코드는 AI를 활용하여 정리된 부분이 있으나, 각 전처리 과정과 알고리즘 선택 이유는 팀원들이 검토하고 이해한 내용을 바탕으로 작성되었다.
