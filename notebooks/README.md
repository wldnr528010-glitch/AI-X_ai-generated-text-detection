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

Essay는 개인적인 경험, 주장, 감정 표현, 연결어 사용이 많이 나타난다.
반면, News는 객관적이고 형식적인 문장 구조를 가지는 경우가 많다.
두 데이터를 합쳐서 하나의 모델로 학습하면, 모델이 AI 여부가 아니라 장르 차이를 학습할 가능성이 있다.

그래서 Essay용 모델과 News용 모델을 따로 학습하고, 두 장르에서 성능이 어떻게 달라지는지 비교하였다.

Hard dataset은 메인 실험과 별도로 진행한 추가 실험이다.
더 어려운 데이터 환경에서 고전 머신러닝 모델과 BERT 기반 모델의 성능 차이를 확인하기 위한 목적으로 사용하였다.

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

이러한 전처리를 적용한 이유는 텍스트 데이터를 모델에 바로 넣을 수는 없기 때문이다.
텍스트에는 대문자, 특수문자, 의미 없는 단어들이 섞여 있어서 그대로 쓰면 모델이 불필요한 패턴을 학습할 수 있다

예를 들어 "The"와 "the"는 같은 단어인데, 전처리를 하지 않으면 모델은 이 둘을 서로 다른 단어로 인식한다
또, "is", "the", "and" 같은 단어는 AI 글이든 사람 글이든 똑같이 많이 나오기 때문에 구분에 도움이 안 된다.

텍스트를 소문자로 통일하고, 특수문자를 없애고, 의미 없는 단어(불용어)를 제거했다.
Lemmatization은 "running"을 "run"으로 바꾸는 것처럼 단어를 원래 형태로 통일해주는 과정인데, 같은 의미의 단어가 다르게 카운트되는 걸 막기 위해 적용했다.

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

모델을 하나만 사용하면 성능이 높게 나왔을 때 모델이 잘 학습된 건지, 아니면 원래 구분하기 쉬운 데이터였는지 판단하기 어렵다.
그래서 성격이 다른 여러 모델을 함께 비교했다.

본 프로젝트에서는 텍스트 데이터를 바로 모델에 입력할 수 없기 때문에,
먼저 TF-IDF를 사용하여 문장을 숫자 벡터로 변환하였다.

TF-IDF(Term Frequency-Inverse Document Frequency)는 단어의 중요도를 수치로 표현하는 방법이다.
단순히 단어가 몇 번 나왔는지만 세는 게 아니라, 특정 문서에서는 자주 나오지만 전체 문서에서는 흔하지 않은 단어일수록 높은 점수를 부여한다.
예를 들어 AI 텍스트에서 유독 자주 등장하는 단어가 있다면, 그 단어에 높은 가중치가 붙어 모델이 AI 글의 특징으로 인식하게 된다.

사용한 모델은 다음과 같다.

```text
1. Logistic Regression
2. Naive Bayes
3. Random Forest
4. XGBoost
```

### Logistic Regression

각 단어의 TF-IDF 점수에 가중치를 곱해 더한 뒤, 그 값을 0~1 사이의 확률로 변환해 분류하는 방식이다.
예를 들어 결과값이 0.8이면 AI 텍스트일 확률이 80%라고 판단한다.
텍스트 분류에서 가장 기본적으로 사용되는 모델이라, 다른 모델들과 비교하기 위한 기준점(baseline)으로 사용하였다.

### Naive Bayes

각 단어가 AI 글에서 등장할 확률과 Human 글에서 등장할 확률을 각각 계산해서 분류하는 방식이다.
"이 단어들이 동시에 나왔을 때 AI 글일 가능성이 더 높은가?"를 확률적으로 판단한다.
모든 단어가 서로 독립적이라는 가정을 쓰기 때문에 실제와 완전히 맞지는 않지만,
단어 빈도 기반인 TF-IDF와 궁합이 좋고 텍스트 분류에서 오랫동안 검증된 모델이라 baseline과 함께 비교하기 위해 포함하였다.

### Random Forest

여러 개의 결정 트리를 독립적으로 만들고, 각 트리의 예측 결과를 모아 다수결로 최종 답을 내리는 방식이다.
한 개의 트리만 쓰면 데이터에 지나치게 맞춰지는 과적합 문제가 생길 수 있는데, 여러 트리를 합치면 이 문제를 줄일 수 있다.
단순한 선형 모델로는 잡기 어려운 복잡한 패턴도 학습할 수 있는지 확인하기 위해 사용하였다.

### XGBoost

트리를 하나씩 순서대로 만들면서, 이전 트리가 틀린 부분을 다음 트리가 집중해서 보완하는 방식이다.
Random Forest가 트리를 동시에 독립적으로 만드는 것과 달리, XGBoost는 순차적으로 실수를 줄여나가기 때문에 일반적으로 성능이 더 높게 나온다.
앙상블 모델 중에서 가장 높은 성능을 기대할 수 있어 함께 비교하였다.

---

각 모델을 사용한 이유를 정리하면 다음과 같다.

| Model               | Reason                                                                      |
| ------------------- | --------------------------------------------------------------------------- |
| Logistic Regression | 텍스트 분류에서 자주 사용되는 기본적인 선형 분류 모델이므로 baseline으로 사용                            |
| Naive Bayes         | 단어의 등장 확률을 기반으로 분류하는 모델로, 텍스트 분류에서 빠르고 간단하게 적용 가능                         |
| Random Forest       | 여러 결정 트리를 결합하여 복잡한 패턴을 학습할 수 있는지 확인하기 위해 사용                                |
| XGBoost             | 강력한 앙상블 모델로, 고전 머신러닝 모델 중 높은 성능을 보일 수 있는지 확인하기 위해 사용                       |

결과적으로 Essay와 News 데이터에서는 네 모델 모두 비슷하게 높은 성능이 나왔는데,
이를 보면 이 데이터는 단어 패턴만으로도 충분히 AI 글과 사람 글을 구분할 수 있는 수준이라는 것을 알 수 있다.


고전 모델의 성능 비교 이미지는 다음 경로에 저장된다.

```text
../results/figures/essay_classical.png
../results/figures/news_classical.png
../results/figures/hard_classical.png
```

---

## 7. BERT-based Model

고전 머신러닝 모델은 결국 단어가 몇 번 나왔는지를 보는 방식이다.
그런데 같은 단어라도 앞뒤 문맥에 따라 의미가 달라질 수 있다.
예를 들어 "나는 사과했다"에서 사과가 과일인지 사죄인지는 앞뒤 문장을 봐야 알 수 있다.
고전 모델은 이걸 구분하지 못하지만, BERT는 문맥을 함께 읽어서 의미를 파악할 수 있다.

그래서 단어 빈도만 보는 고전 모델이랑 문맥까지 보는 BERT가 성능 차이가 얼마나 나는지 비교해보고 싶었다.

BERT는 문장 전체의 흐름을 양방향으로 읽어서 단어의 문맥까지 반영할 수 있는 모델이다.
본 프로젝트에서는 실행 시간을 고려해 BERT의 경량화 버전인 DistilBERT를 사용하였다.
DistilBERT는 BERT보다 가볍고 빠르게 학습할 수 있으면서도 문맥 정보를 반영하는 능력은 충분히 유지하고 있어 이번 프로젝트에 적합하다고 판단하였다.

실제로 Essay, News처럼 비교적 쉬운 데이터에서는 고전 모델과 성능 차이가 크지 않았는데,
Hard dataset처럼 구분하기 어려운 데이터에서는 BERT가 훨씬 높은 성능을 보였다.
데이터가 어려울수록 단순한 단어 빈도보다 문맥 이해가 더 중요해진다는 걸 이번 실험을 통해 확인할 수 있다.

---

## 8. Evaluation Metrics

모델을 학습시킨 뒤에는 "이 모델이 얼마나 잘 작동하는가"를 숫자로 확인해야 한다.
이때 사용하는 게 평가 지표인데, 본 프로젝트에서는 Accuracy와 F1-score를 함께 사용하였다.

### Accuracy

Accuracy는 전체 데이터 중에서 모델이 올바르게 예측한 비율이다.
예를 들어 100개의 텍스트 중 95개를 AI/Human으로 정확히 분류했다면 Accuracy는 95%가 된다.
직관적으로 이해하기 쉽고, 전체적인 성능을 한눈에 볼 수 있다는 장점이 있다.

하지만 Accuracy에는 한계가 있다.
만약 데이터의 90%가 AI 텍스트라면, 모델이 모든 텍스트를 AI로 분류하더라도 Accuracy는 90%가 나온다.
숫자만 보면 잘 작동하는 것처럼 보이지만, 사실 Human 텍스트를 전혀 잡아내지 못하는 모델인 것이다.

### F1-score

F1-score는 이런 Accuracy의 한계를 보완하기 위한 지표다.
F1-score를 이해하려면 먼저 Precision과 Recall을 알아야 한다.

- **Precision**: 모델이 AI라고 예측한 것 중에서 실제로 AI인 비율
- **Recall**: 실제 AI 텍스트 중에서 모델이 AI라고 제대로 잡아낸 비율

F1-score는 이 두 값의 균형을 하나의 숫자로 나타낸 것이다.
Precision이 높아도 Recall이 낮으면, 또는 그 반대여도 F1-score는 낮게 나오기 때문에, 모델이 한쪽으로 치우쳐 예측하는 상황을 잡아낼 수 있다.

### 왜 두 지표를 함께 사용했나?

Accuracy는 전체적인 성능을 보여주지만 데이터 구성에 따라 왜곡될 수 있고,
F1-score는 AI와 Human 텍스트를 균형 있게 잘 분류하는지를 보여준다.
두 지표를 함께 보면 서로의 단점을 보완할 수 있다.

우리 데이터는 Human:AI = 1:1로 균형 잡혀 있어서 Accuracy가 크게 왜곡될 가능성은 낮다.
그럼에도 두 지표를 함께 사용한 이유는, 모델이 특정 클래스에 치우쳐 예측하는 상황을 놓치지 않기 위해서다.
Accuracy만 보고 "잘 됐다"고 판단하기보다, F1-score까지 함께 확인해야 모델 성능을 더 정확하게 평가할 수 있다고 판단하였다.

| Metric   | Meaning                            |
| -------- | ---------------------------------- |
| Accuracy | 전체 데이터 중 모델이 올바르게 예측한 비율         |
| F1-score | Precision과 Recall의 균형을 고려한 성능 지표 |


---

## 9. Main Experiment and Additional Experiment

본 프로젝트의 실험은 크게 두 가지로 나눴다.

```text
Main Experiment:
- Essay dataset
- News dataset

Additional Experiment:
- Hard dataset
```

Essay와 News는 이 프로젝트의 핵심 데이터셋이다.
두 장르의 텍스트에서 AI 탐지 성능이 어떻게 다른지 비교하는 게 주된 목적이었기 때문에, 이 두 데이터셋을 중심으로 실험을 설계하였다.

Hard dataset은 메인 실험과 별도로 진행한 추가 실험이다.
더 어렵고 복잡한 텍스트 환경에서도 모델이 잘 작동하는지 확인하고 싶었고, 특히 이 조건에서 고전 모델과 BERT의 성능 차이가 얼마나 벌어지는지 보고 싶었다.

결과적으로 쉬운 데이터(Essay, News)에서는 고전 모델도 충분히 잘 작동했지만,
Hard dataset에서는 BERT가 눈에 띄게 높은 성능을 보였다.
데이터 난이도에 따라 적합한 모델이 달라질 수 있다는 점을 확인할 수 있다.

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
