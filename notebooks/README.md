# Notebooks

이 폴더는 본 프로젝트의 핵심 분석 코드를 담고 있다.
프로젝트의 전체 흐름은 `human_vs_ai_text_classification.py` 파일을 중심으로 진행되며,
데이터 전처리부터 시각화, 모델 학습, 최종 성능 비교까지 하나의 코드 안에서 순서대로 실행된다.

본 프로젝트는 생성형 AI가 작성한 텍스트와 사람이 작성한 텍스트를 구분하는 AI-generated text detection 문제를 다룬다.
특히 Essay와 News 데이터를 main experiment로 사용하고, Hard dataset을 additional experiment로 활용하여
데이터의 장르와 난이도에 따라 모델 성능이 어떻게 달라지는지 확인하였다.

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

Hard dataset은 main experiment가 아니라 additional experiment로 사용하였다.  
더 어려운 데이터 환경에서 고전 머신러닝 모델과 BERT 기반 모델의 성능 차이를 확인하기 위한 목적으로 사용하였다.


각 단계별 내용을 요약하면 다음과 같다.

| 단계 | 내용 |
|---|---|
| Step 1. 라이브러리 불러오기 | 분석에 필요한 모든 도구를 한 번에 준비한다. |
| Step 2. 전처리 함수 정의 | 텍스트를 정리하는 전처리 clean_text() 함수를 만든다. |
| Step 3. 데이터 불러오기 | Essay, News, Hard 원본 데이터를 가져온다. |
| Step 4. 전처리 적용 | Step 2에서 만든 함수를 실제 데이터에 적용해 clean_text 컬럼을 생성한다. |
| Step 5. 단어 빈도 시각화 | 사람 글과 AI 글에서 자주 나오는 단어를 그래프로 비교한다. |
| Step 6. 고전 모델 학습 | TF-IDF로 텍스트를 숫자로 변환 후 4가지 모델을 학습한다. |
| Step 7. 최신 모델 학습 | 문맥을 이해하는 DistilBERT 모델을 학습한다. |
| Step 8. 성능 비교 | 모든 모델의 Accuracy와 F1-score를 비교한다. |
| Step 9. 결과 저장 | 최종 결과 이미지와 CSV 파일을 저장한다. |

---


## Step 1. 라이브러리 불러오기

분석에 필요한 모든 라이브러리를 한 번에 불러오는 코드이다.

```python
import pandas as pd                       # 엑셀처럼 데이터를 표로 다루는 도구
import re                                 # 텍스트에서 특수문자 찾아 지우는 도구
import numpy as np                        # 숫자 계산을 빠르게 해주는 도구
import matplotlib.pyplot as plt           # 막대그래프, 선그래프 그리는 도구
from collections import Counter           # 단어 개수를 세는 도구
from wordcloud import WordCloud           # 단어 구름 그림 만드는 도구

import nltk
from nltk.corpus import stopwords         # 'the', 'is' 같은 불필요한 단어 목록
from nltk.stem import WordNetLemmatizer   # 단어를 원형으로 바꿔주는 도구 (running → run)

from sklearn.feature_extraction.text import TfidfVectorizer    # 글을 숫자로 변환
from sklearn.model_selection import train_test_split           # 데이터를 학습/테스트로 분리
from sklearn.linear_model import LogisticRegression            # (고전 분류모델) 분류모델 1 (LogisticRegression)
from sklearn.naive_bayes import MultinomialNB                  # (고전 분류모델) 분류모델 2 (naive_bayes)
from sklearn.ensemble import RandomForestClassifier            # (고전 분류모델) 분류모델 3 (RandomForestClassifier)
from xgboost import XGBClassifier                              # (고전 분류모델) 분류모델 4 (XGBoost)
from sklearn.metrics import accuracy_score, f1_score           # 분류모델 성능 측정 도구

import torch                                                                             # AI 딥러닝 연산 도구
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification        # (최신 분류 모델) BERT 모델 불러오는 도구
from torch.utils.data import Dataset, DataLoader                                         # 데이터를 배치로 나눠주는 도구
from torch.optim import AdamW                                                            # 모델 학습 최적화 도구

```

---

## Step 2. 텍스트 전처리 함수 정의


텍스트 데이터는 원본 상태 그대로 모델에 입력하기 어렵기 때문에, `clean_text()` 함수를 통해 전처리를 수행하였다.

전처리 과정은 다음과 같다.

```
1. 이상기호 일반 문자 변환
2. 영어 외 문자 제거
3. 소문자 변환
4. 알파벳과 공백만 남기기
5. 불용어 제거
6. Lemmatization
7. 너무 짧은 텍스트 제거 (Step 4에서 별도 처리)
```

이러한 전처리를 적용한 이유는 텍스트 데이터를 모델에 바로 넣을 수는 없기 때문이다.
텍스트에는 대문자, 특수문자, 의미 없는 단어들이 섞여 있어서 그대로 쓰면 모델이 불필요한 패턴을 학습할 수 있다.

예를 들어 "The"와 "the"는 같은 단어인데, 전처리를 하지 않으면 모델은 이 둘을 서로 다른 단어로 인식한다.
또, "is", "the", "and" 같은 단어는 AI 글이든 사람 글이든 똑같이 많이 나오기 때문에 구분에 도움이 되지 않는다.

텍스트를 소문자로 통일하고, 특수문자를 없애고, 의미 없는 단어(불용어)를 제거했다.
Lemmatization은 "running"을 "run"으로 바꾸는 것처럼 단어를 원래 형태로 통일해주는 과정인데, 같은 의미의 단어가 다르게 카운트되는 걸 막기 위해 적용했다.


텍스트 데이터를 모델 학습에 적합한 형태로 정리하는 clean_text() 함수를 정의하는 코드이다.
이 단계에서는 함수를 정의하기만 하며, 실제 데이터에 적용하는 것은 Step 4에서 수행한다.
```python
def clean_text(text):                  
    if not isinstance(text, str):
        return ""

    # 1. 이상기호 일반 문자 변환
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    text = text.replace('\u00A0', ' ').replace('\u2026', '...')

    # 2. 영어 외 문자 제거
    text = re.sub(r'[^\x00-\x7F]', '', text)

    # 3. 소문자 변환
    text = text.lower()

    # 4. 알파벳과 공백만 남기기
    text = re.sub(r'[^a-z\s]', '', text)

    # 5. 불용어 제거 / 6. Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(tokens)
```


| 전처리 단계 | 적용 이유 |
|---|---|
| 이상기호 일반 문자 변환 | 따옴표, 대시 등 특수 문자가 모델 오류를 유발할 수 있어 일반 문자로 변환 |
| 영어 외 문자 제거 | 한글, 이모지 등 영어 텍스트 분석에 불필요한 문자 제거 |
| 소문자 변환 | "The"와 "the"를 같은 단어로 처리하기 위해 |
| 알파벳과 공백만 남기기 | 숫자, 기호 등 텍스트 분류와 무관한 문자 제거 |
| 불용어 제거 | "the", "is" 등 분류에 도움이 되지 않는 단어 제거 |
| Lemmatization | "running"→"run"으로 통일해 중복 카운트 방지 |
| 너무 짧은 텍스트 제거 | 단어 수가 너무 적으면 패턴 학습이 불가능하므로, Step 4에서 별도 처리 |

---

## Step 3. Essay, News, Hard dataset 불러오기

코드는 프로젝트 최상위 폴더를 기준으로 실행한다고 가정한다.
사용하는 원본 데이터 경로는 다음과 같다.

```text
../data/project_dataset/essay_dataset.csv
../data/project_dataset/news_dataset.csv
../data/project_dataset/hard_dataset.csv
```

| Label | Meaning |
|---|---|
| 0 | Human-written text |
| 1 | AI-generated text |

---

## Step 4. 데이터 전처리 및 clean_text 생성

Step 2에서 정의한 `clean_text()` 함수를 각 데이터셋에 적용하여 전처리된 텍스트를 `clean_text` 컬럼에 저장한다.


불러온 데이터에서 빈 칸이나 이상한 값을 제거하고, STEP 2.에서 정의한 전처리 함수를 적용해 전처리된 텍스트를 clean_text 컬럼에 저장하는 코드이다.
```python
# 불용어 목록과 Lemmatizer 초기화 (clean_text 함수 실행 전에 반드시 필요)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 에세이 데이터 전처리
df_essay['generated'] = pd.to_numeric(df_essay['generated'], errors='coerce')
# generated 컬럼의 값을 숫자로 변환한다. 숫자로 바꿀 수 없는 값은 빈 칸으로 처리한다

df_essay = df_essay.dropna(subset=['generated', 'text'])
# generated나 text 중 하나라도 빈 칸인 행을 삭제한다

df_essay['generated'] = df_essay['generated'].astype(int)
# 0.0, 1.0 처럼 소수점 형태로 저장된 숫자를 0, 1 정수로 변환한다

df_essay = df_essay[df_essay['generated'].isin([0, 1])].reset_index(drop=True)
# 0(사람 글) 또는 1(AI 글)이 아닌 이상한 값을 가진 행을 제거한다

df_essay['clean_text'] = df_essay['text'].apply(clean_text)
# Step 2에서 만든 clean_text() 함수를 모든 행에 적용해 전처리된 텍스트를 저장한다

df_essay = df_essay[df_essay['clean_text'].str.split().str.len() >= 5].reset_index(drop=True)
# 전처리 후 단어가 5개보다 적게 남은 행을 삭제한다. 너무 짧으면 모델이 학습할 정보가 없기 때문이다
```

뉴스 데이터와 Hard dataset도 동일한 과정으로 전처리를 수행하였다.

전처리가 완료된 데이터를 CSV파일 형식으로 저장하는 코드이다.
```python
# 전처리된 데이터 CSV로 저장
df_essay.to_csv('essay_clean_dataset.csv', index=False, encoding='utf-8')
df_news.to_csv('news_clean_dataset.csv',   index=False, encoding='utf-8')
df_hard.to_csv('hard_clean_dataset.csv',   index=False, encoding='utf-8')
```

전처리가 완료된 데이터는 아래 경로에 CSV로 저장된다.
```
../data/processed/essay_clean_dataset.csv
../data/processed/news_clean_dataset.csv
../data/processed/hard_clean_dataset.csv
```

---

## Step 5. Human text와 AI text의 단어 빈도 시각화

전처리된 텍스트를 바탕으로 Human-written text와 AI-generated text에서
자주 등장하는 단어를 비교하였다.
이 분석은 모델 학습 전에 사람 글과 AI 글 사이에 단어 사용 패턴 차이가 존재하는지
시각적으로 먼저 확인하기 위한 과정이다.


사람 글과 AI 글에서 자주 등장하는 단어 상위 20개를 막대그래프와 WordCloud로 시각화하는 코드이다.
```python
def visualize_words(df, title):

    # 사람 글과 AI 글의 단어를 각각 추출
    human_words = ' '.join(df[df['generated']==0]['clean_text'].dropna()).split()
    ai_words    = ' '.join(df[df['generated']==1]['clean_text'].dropna()).split()

    # 상위 20개 단어 추출
    human_top = Counter(human_words).most_common(20)
    ai_top    = Counter(ai_words).most_common(20)

    # 막대그래프 + WordCloud 시각화 (2x2 구성)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{title} - Word Frequency Analysis', fontsize=16, fontweight='bold')
    ...
    plt.savefig(f'../results/figures/{title.lower()}_words.png', dpi=150, bbox_inches='tight')
```

각 데이터셋에 대해 다음 4가지 시각화를 생성한다.

- **Human Top 20 Words** : 사람 글에서 가장 많이 나온 단어 20개 막대그래프
- **AI Top 20 Words** : AI 글에서 가장 많이 나온 단어 20개 막대그래프
- **Human WordCloud** : 사람 글의 단어 빈도를 구름 형태로 시각화
- **AI WordCloud** : AI 글의 단어 빈도를 구름 형태로 시각화

생성되는 이미지 파일은 다음과 같다.

```
../results/figures/essay_words.png
../results/figures/news_words.png
../results/figures/hard_words.png
```

세 데이터셋의 단어 빈도를 시각화한 결과, AI 텍스트와 Human 텍스트 사이에
몇 가지 뚜렷한 차이가 나타났다.

Essay 데이터에서는 Human 텍스트에 "would", "vote", "president", "get" 같은
개인 의견이나 행동을 나타내는 단어가 많이 등장한 반면,
AI 텍스트에서는 "student", "community", "individual", "important", "example" 같은
보다 일반적이고 격식체적인 단어가 상위권에 많이 나타났다.

News 데이터에서는 Human 텍스트에 "trump", "data", "said", "site" 같은
구체적인 사실을 나타내는 단어가 많이 등장한 반면,
AI 텍스트에서는 "community", "health", "significant", "challenge", "political" 같은
보다 포괄적이고 추상적인 단어가 많이 나타났다.

Hard 데이터에서는 Human과 AI의 상위 단어가 많이 겹쳤지만,
AI 텍스트에서 "also", "community", "may", "life" 같은 단어의 비중이 상대적으로 높았고,
Human 텍스트에서는 "would", "vote", "said", "data" 같은 단어가 더 많이 등장했다.

전반적으로 Human 텍스트는 구체적이고 다양한 단어를 사용하는 경향이 있었고,
AI 텍스트는 일반적이고 포괄적인 단어를 반복적으로 사용하는 경향이 있었다.
이를 통해 단어 사용 패턴만으로도 AI 글과 사람 글을 어느 정도 구분할 수 있다는
가능성을 확인하였다.

---

## Step 6. TF-IDF 기반 고전 머신러닝 모델 학습

모델을 하나만 사용하면 성능이 높게 나왔을 때 모델이 잘 학습된 건지,
아니면 원래 구분하기 쉬운 데이터였는지 판단하기 어렵다.
따라서 성격이 다른 여러 모델을 함께 비교하였다.

본 프로젝트에서는 텍스트 데이터를 바로 모델에 입력할 수 없기 때문에,
먼저 TF-IDF를 사용하여 문장을 숫자 벡터로 변환하였다.

TF-IDF(Term Frequency-Inverse Document Frequency)는 단어의 중요도를 수치로 표현하는 방법이다.
단순히 단어가 몇 번 나왔는지만 세는 게 아니라,
특정 문서에서는 자주 나오지만 전체 문서에서는 흔하지 않은 단어일수록 높은 점수를 부여한다.


텍스트를 숫자로 바꾸는 코드이다. 단어가 얼마나 자주, 얼마나 특별하게 쓰였는지를 점수로 표현한다.
```python
# max_features=10000: 가장 많이 등장하는 단어 10,000개만 사용
# ngram_range=(1,2): 단어 1개짜리와 2개짜리 조합을 모두 고려
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = tfidf.fit_transform(df['clean_text'])
y = df['generated']
```

전체 데이터의 80%로 학습하고, 나머지 20%로 테스트하는 코드이다.
```python
# 전체 데이터의 80%로 학습, 나머지 20%로 테스트
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Logistic Regression

각 단어의 TF-IDF 점수에 가중치를 곱해 더한 뒤, 그 값을 0~1 사이의 확률로 변환해 분류하는 방식이다.
예를 들어 결과값이 0.8이면 AI 텍스트일 확률이 80%라고 판단한다.
텍스트 분류에서 가장 기본적으로 사용되는 모델이라, 다른 모델들과 비교하기 위한 기준점(baseline)으로 사용하였다.

```python
model = LogisticRegression(max_iter=1000)
# max_iter=1000: 최적의 가중치를 찾기 위해 최대 1000번 반복
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Naive Bayes

각 단어가 AI 글에서 등장할 확률과 Human 글에서 등장할 확률을 각각 계산해서 분류하는 방식이다.
"이 단어들이 동시에 나왔을 때 AI 글일 가능성이 더 높은가?"를 확률적으로 판단한다.
단어 빈도 기반인 TF-IDF와 궁합이 좋고, 텍스트 분류에서 오랫동안 검증된 모델이라 포함하였다.

```python
model = MultinomialNB()
# MultinomialNB: 단어 등장 횟수(빈도) 기반으로 확률을 계산하는 Naive Bayes
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Random Forest

여러 개의 결정 트리를 독립적으로 만들고,
각 트리의 예측 결과를 모아 다수결로 최종 답을 내리는 방식이다.
한 개의 트리만 쓰면 데이터에 지나치게 맞춰지는 과적합 문제가 생길 수 있는데,
여러 트리를 합치면 이 문제를 줄일 수 있다.

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
# n_estimators=100: 결정 트리 100개를 만들어 앙상블
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### XGBoost

트리를 하나씩 순서대로 만들면서,
이전 트리가 틀린 부분을 다음 트리가 집중해서 보완하는 방식이다.
Random Forest가 트리를 동시에 독립적으로 만드는 것과 달리,
XGBoost는 순차적으로 실수를 줄여나가기 때문에 일반적으로 성능이 더 높게 나온다.

```python
model = XGBClassifier(eval_metric='logloss', random_state=42)
# eval_metric='logloss': 이진 분류에서 사용하는 손실 함수
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

각 모델을 사용한 이유를 정리하면 다음과 같다.

| Model | Reason |
|---|---|
| Logistic Regression | 텍스트 분류에서 자주 사용되는 기본적인 선형 분류 모델이므로 baseline으로 사용 |
| Naive Bayes | 단어의 등장 확률을 기반으로 분류하는 모델로, 텍스트 분류에서 빠르고 간단하게 적용 가능 |
| Random Forest | 여러 결정 트리를 결합하여 복잡한 패턴을 학습할 수 있는지 확인하기 위해 사용 |
| XGBoost | 강력한 앙상블 모델로, 고전 머신러닝 모델 중 높은 성능을 보일 수 있는지 확인하기 위해 사용 |

결과적으로 Essay와 News 데이터에서는 네 모델 모두 비슷하게 높은 성능이 나왔는데,
이를 보면 이 데이터는 단어 패턴만으로도 충분히 AI 글과 사람 글을 구분할 수 있는 수준이라는 것을 알 수 있다.

고전 모델의 성능 비교 이미지는 다음 경로에 저장된다.

```text
../results/figures/essay_classical.png
../results/figures/news_classical.png
../results/figures/hard_classical.png
```

---

## Step 7. DistilBERT 기반 모델 학습

고전 머신러닝 모델은 결국 단어가 몇 번 나왔는지를 보는 방식이다.
그런데 같은 단어라도 앞뒤 문맥에 따라 의미가 달라질 수 있다.
예를 들어 "나는 사과했다"에서 사과가 과일인지 사죄인지는
앞뒤 문장을 봐야 알 수 있다.
고전 모델은 이걸 구분하지 못하지만, BERT는 문맥을 함께 읽어서 의미를 파악할 수 있다.

그래서 단어 빈도만 보는 고전 모델이랑 문맥까지 보는 BERT가
성능 차이가 얼마나 나는지 비교하였다.

BERT는 문장 전체의 흐름을 양방향으로 읽어서 단어의 문맥까지 반영할 수 있는 모델이다.
본 프로젝트에서는 실행 시간을 고려해 BERT의 경량화 버전인 DistilBERT를 사용하였다.
DistilBERT는 BERT보다 가볍고 빠르게 학습할 수 있으면서도
문맥 정보를 반영하는 능력은 충분히 유지하고 있어 이번 프로젝트에 적합하다고 판단하였다.


BERT는 텍스트를 그대로 이해하지 못하고 숫자만 읽을 수 있다.
이 코드는 단어를 숫자로 바꿔주는 번역기(토크나이저)를 준비하고, 모든 문장을 128개 단어 길이로 맞춰주는 코드이다.
128개보다 길면 잘라내고, 짧으면 빈 칸을 채워 길이를 통일한다.
```python
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# distilbert-base-uncased: 대소문자 구분 없이 학습된 경량화 BERT 모델

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_len=128):
        # max_len=128: 한 문장에서 최대 128개의 토큰만 사용
        # 128개를 초과하는 부분은 잘라내고, 128개보다 짧으면 padding으로 채운다
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)
```


BERT 모델을 불러온 뒤, 문제를 풀고 틀린 부분을 확인하며 수정하는 과정을 여러 번 반복하여 AI 글과 사람 글을 구분하는 능력을 키우는 코드이다.
```python
# DistilBERT 모델 불러오기 (이진 분류이므로 num_labels=2)
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
).to(device)

# AdamW: BERT 계열 모델 학습에 주로 사용하는 최적화 알고리즘
# lr=2e-5: BERT 계열 모델에서 일반적으로 권장되는 학습률
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()    # 이전 gradient 초기화
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device)
        )
        outputs.loss.backward()  # 역전파로 gradient 계산
        optimizer.step()         # 가중치 업데이트
```

실제로 Essay, News처럼 비교적 쉬운 데이터에서는 고전 모델과 성능 차이가 크지 않았는데,
Hard dataset처럼 구분하기 어려운 데이터에서는 BERT가 훨씬 높은 성능을 보였다.
데이터가 어려울수록 단순한 단어 빈도보다 문맥 이해가 더 중요해진다는 걸 확인할 수 있다.

---

## Step 8. 전체 모델 성능 비교

모델을 학습시킨 뒤에는 "이 모델이 얼마나 잘 작동하는가"를 숫자로 확인해야 한다.
본 프로젝트에서는 Accuracy와 F1-score를 함께 사용하였다.


### Accuracy

Accuracy는 전체 데이터 중에서 모델이 올바르게 예측한 비율이다.
예를 들어 100개의 텍스트 중 95개를 AI/Human으로 정확히 분류했다면 Accuracy는 95%가 된다.

하지만 Accuracy에는 한계가 있다.
만약 데이터의 90%가 AI 텍스트라면, 모델이 모든 텍스트를 AI로 분류하더라도 Accuracy는 90%가 나온다.
숫자만 보면 잘 작동하는 것처럼 보이지만, 사실 Human 텍스트를 전혀 잡아내지 못하는 모델인 것이다.

### F1-score

F1-score는 이런 Accuracy의 한계를 보완하기 위한 지표다.
F1-score를 이해하려면 먼저 Precision과 Recall을 알아야 한다.

- **Precision**: 모델이 AI라고 예측한 것 중에서 실제로 AI인 비율
- **Recall**: 실제 AI 텍스트 중에서 모델이 AI라고 제대로 잡아낸 비율

F1-score는 이 두 값의 균형을 하나의 숫자로 나타낸 것이다.
Precision이 높아도 Recall이 낮으면, 또는 그 반대여도 F1-score는 낮게 나오기 때문에,
모델이 한쪽으로 치우쳐 예측하는 상황을 잡아낼 수 있다.

### 왜 두 지표를 함께 사용했나?

본 프로젝트의 데이터는 Human:AI = 1:1로 균형 잡혀 있어서 Accuracy가 크게 왜곡될 가능성은 낮다.
그럼에도 두 지표를 함께 사용한 이유는, 모델이 특정 클래스에 치우쳐 예측하는 상황을 놓치지 않기 위해서다.
Accuracy만 보고 "잘 됐다"고 판단하기보다, F1-score까지 함께 확인해야 모델 성능을 더 정확하게 평가할 수 있다고 판단하였다.


학습이 끝난 모델이 테스트 데이터를 얼마나 정확하게 분류했는지 Accuracy와 F1-score로 측정하는 코드이다.
```python
from sklearn.metrics import accuracy_score, f1_score

acc = accuracy_score(y_test, y_pred)  # 전체 중 맞게 예측한 비율
f1  = f1_score(y_test, y_pred)        # Precision과 Recall의 균형 점수
```

| Metric | Meaning |
|---|---|
| Accuracy | 전체 데이터 중 모델이 올바르게 예측한 비율 |
| F1-score | Precision과 Recall의 균형을 고려한 성능 지표 |

세 가지 데이터셋에서의 최종 결과는 다음과 같다.

```
# ======================================================================
# 모델                     Essay     News      Hard
# ----------------------------------------------------------------------
# Logistic Regression     97.6%    99.1%     87.6%
# Naive Bayes             95.9%    93.7%     82.2%
# Random Forest           97.7%    97.7%     86.9%
# XGBoost                 97.8%    99.2%     85.9%
# BERT                    97.7%    99.2%     93.3%
# ======================================================================
```

쉬운 데이터(Essay, News)에서는 고전 모델도 충분히 잘 작동했지만,
Hard dataset에서는 BERT가 더 높은 성능을 보였다.
데이터 난이도에 따라 적합한 모델이 달라질 수 있다는 점을 확인할 수 있다.

---

## Step 9. 결과 이미지 및 CSV 저장

코드를 실행하면 다음 결과물이 생성된다.

```python
# 최종 결과 CSV 저장
df_results.to_csv('final_results.csv', index=False, encoding='utf-8-sig')
```

```
Step 4 에서 저장
../data/processed/
├── essay_clean_dataset.csv
├── news_clean_dataset.csv
└── hard_clean_dataset.csv

Step 5,6 에서 저장
../results/figures/
├── essay_words.png
├── news_words.png
├── hard_words.png
├── essay_classical.png
├── news_classical.png
├── hard_classical.png
└── final_all_models.png

Step 9 에서 저장
../results/tables/
└── final_results.csv
```

---

## How to Run

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

## Notes

- 일부 데이터 파일은 용량 문제로 GitHub에 업로드되지 않을 수 있다.
- 업로드되지 않은 전처리 데이터는 코드를 실행하면 다시 생성할 수 있다.
- BERT 학습은 실행 환경에 따라 시간이 오래 걸릴 수 있다.
- GPU가 없는 경우 CPU로 실행되며, 학습 시간이 증가할 수 있다.
- 본 코드는 AI를 활용하여 정리된 부분이 있으나, 각 전처리 과정과 알고리즘 선택 이유는 팀원들이 검토하고 이해한 내용을 바탕으로 작성되었다.
