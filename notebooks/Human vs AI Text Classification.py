# ============================================================
# AI가 쓴 글인지 사람이 쓴 글인지 컴퓨터가 맞히게 만들기
# AI-Generated Text Detection Project
# ============================================================


## 1. 코드 실행을 위한 필요 프로그램을 설치
### (해당 코드는 프로젝트 전체에서 쓸 도구들을 한꺼번에 준비하는 코드입니다. 요리 전에 재료와 도구를 꺼내는 것과 같습니다.)


import sys
!{sys.executable} -m pip install nltk xgboost wordcloud matplotlib scikit-learn torch transformers --quiet   # 프로그램 설치하기

import pandas as pd                     # 엑셀처럼 데이터를 표로 다루는 도구
import re                               # 텍스트에서 특수문자 찾아 지우는 도구
import numpy as np                      # 숫자 계산을 빠르게 해주는 도구
import matplotlib.pyplot as plt         # 막대그래프, 선그래프 그리는 도구
from collections import Counter         # 단어 개수를 세는 도구
from wordcloud import WordCloud         # 단어 구름 그림 만드는 도구

import nltk                                 # 영어 텍스트 분석 전문 도구
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords           # 'the', 'is' 같은 불필요한 단어 목록
from nltk.stem import WordNetLemmatizer     # 단어를 원형으로 바꿔주는 도구 ex) running → run

from sklearn.feature_extraction.text import TfidfVectorizer       # 글을 숫자로 변환
from sklearn.model_selection import train_test_split              # 데이터를 학습/테스트로 분리
from sklearn.linear_model import LogisticRegression               # (고전 분류모델) 분류모델 1 (LogisticRegression)
from sklearn.naive_bayes import MultinomialNB                     # (고전 분류모델) 분류모델 2 (naive_bayes)
from sklearn.ensemble import RandomForestClassifier               # (고전 분류모델) 분류모델 3 (RandomForestClassifier)
from xgboost import XGBClassifier                                 # (고전 분류모델) 분류모델 4 (XGBoost)
from sklearn.metrics import accuracy_score, f1_score              # 분류모델 성능 측정 도구

import torch                                                                             # AI 딥러닝 연산 도구
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification        # (최신 분류 모델) BERT 모델 불러오는 도구
from torch.utils.data import Dataset, DataLoader                                         # 데이터를 배치로 나눠주는 도구
from torch.optim import AdamW                                                            # 모델 학습 최적화 도구

print("✅ 모든 라이브러리 로드 완료!")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                    # GPU(그래픽카드)가 있으면 GPU를, 없으면 CPU를 사용한다.
print(f"✅ 사용 디바이스: {device}")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #




## 2. 데이터 파일 전처리 함수 정의: 컴퓨터가 읽기 좋게 글을 정리할 수 있는 함수 정의(특수문자, 소문자 변환, 필요없는 단어 삭제)


def clean_text(text):                  
    if not isinstance(text, str):
        return ""
    # 깨진 문자 치환
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    text = text.replace('\u00A0', ' ').replace('\u2026', '...')
    # ASCII 범위 밖 문자 제거
    text = re.sub(r'[^\x00-\x7F]', '', text)
    # 소문자 변환
    text = text.lower()
    # 알파벳과 공백만 남기기
    text = re.sub(r'[^a-z\s]', '', text)
    # 불용어 제거 + 어간 추출
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(tokens)

print("✅ 전처리 함수 정의 완료!")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #




## 3. 데이터를 가져와서 전처리 실행(에세이, 뉴스, 어려운 데이터)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 에세이 데이터 전처리
df_essay = pd.read_csv('essay_dataset.csv', encoding='latin-1')
df_essay['generated'] = pd.to_numeric(df_essay['generated'], errors='coerce')
df_essay = df_essay.dropna(subset=['generated', 'text'])
df_essay['generated'] = df_essay['generated'].astype(int)
df_essay = df_essay[df_essay['generated'].isin([0, 1])].reset_index(drop=True)
df_essay['clean_text'] = df_essay['text'].apply(clean_text)
df_essay = df_essay[df_essay['clean_text'].str.split().str.len() >= 5].reset_index(drop=True)
print(f"📄 에세이 데이터: {len(df_essay)}행")
print(f"   사람(0): {(df_essay['generated']==0).sum()}개 | AI(1): {(df_essay['generated']==1).sum()}개")

# 뉴스 데이터 전처리
df_news = pd.read_csv('news_dataset.csv', encoding='latin-1')
df_news['generated'] = pd.to_numeric(df_news['generated'], errors='coerce')
df_news = df_news.dropna(subset=['generated', 'text'])
df_news['generated'] = df_news['generated'].astype(int)
df_news = df_news[df_news['generated'].isin([0, 1])].reset_index(drop=True)
df_news['clean_text'] = df_news['text'].apply(clean_text)
df_news = df_news[df_news['clean_text'].str.split().str.len() >= 5].reset_index(drop=True)
print(f"📰 뉴스 데이터: {len(df_news)}행")
print(f"   사람(0): {(df_news['generated']==0).sum()}개 | AI(1): {(df_news['generated']==1).sum()}개")

# 어려운 데이터 전처리 (ChatGPT 글 포함)
df_hard = pd.read_csv('hard_dataset.csv', encoding='utf-8')
df_hard['generated'] = pd.to_numeric(df_hard['generated'], errors='coerce')
df_hard = df_hard.dropna(subset=['generated', 'text'])
df_hard['generated'] = df_hard['generated'].astype(int)
df_hard = df_hard[df_hard['generated'].isin([0, 1])].reset_index(drop=True)
df_hard['clean_text'] = df_hard['text'].apply(clean_text)
df_hard = df_hard[df_hard['clean_text'].str.split().str.len() >= 5].reset_index(drop=True)
print(f"🔥 어려운 데이터(ChatGPT): {len(df_hard)}행")
print(f"   사람(0): {(df_hard['generated']==0).sum()}개 | AI(1): {(df_hard['generated']==1).sum()}개")

# 전처리된 데이터 CSV로 저장
df_essay.to_csv('essay_clean_dataset.csv', index=False, encoding='utf-8')
df_news.to_csv('news_clean_dataset.csv',   index=False, encoding='utf-8')
df_hard.to_csv('hard_clean_dataset.csv',   index=False, encoding='utf-8')
print("✅ 전처리된 CSV 저장 완료!")

# 전처리 결과 예시
# 원본: When people are seeking advice they usually ask more than one person for help
# 처리후: people seeking advice usually ask one person help help many way example might

# ------------------------------------------------------------------------------------------------------------------------------------------------ #



## 4. 에세이, 뉴스, 어려운 데이터의 단어 빈도 시각화
### 사람이 쓴 글과 AI가 쓴 글에서 자주 등장하는 단어를 시각화합니다.
#### 이를 통해 사람 글과 AI 글의 단어 패턴 차이를 직관적으로 확인할 수 있습니다.


def visualize_words(df, title):
    """사람 글과 AI 글의 단어 빈도를 시각화합니다."""
    human_words = ' '.join(df[df['generated']==0]['clean_text'].dropna()).split()
    ai_words    = ' '.join(df[df['generated']==1]['clean_text'].dropna()).split()

    print(f"사람 단어 수: {len(human_words):,}  |  AI 단어 수: {len(ai_words):,}")

    human_top = Counter(human_words).most_common(20)
    ai_top    = Counter(ai_words).most_common(20)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{title} - Word Frequency Analysis', fontsize=16, fontweight='bold')

    words, counts = zip(*human_top)
    axes[0][0].barh(words[::-1], counts[::-1], color='steelblue')
    axes[0][0].set_title('Human - Top 20 Words')
    axes[0][0].set_xlabel('Count')

    words, counts = zip(*ai_top)
    axes[0][1].barh(words[::-1], counts[::-1], color='tomato')
    axes[0][1].set_title('AI - Top 20 Words')
    axes[0][1].set_xlabel('Count')

    wc_human = WordCloud(width=800, height=400, background_color='white',
                         colormap='Blues').generate(' '.join(human_words))
    axes[1][0].imshow(wc_human, interpolation='bilinear')
    axes[1][0].axis('off')
    axes[1][0].set_title('Human - WordCloud')

    wc_ai = WordCloud(width=800, height=400, background_color='white',
                      colormap='Reds').generate(' '.join(ai_words))
    axes[1][1].imshow(wc_ai, interpolation='bilinear')
    axes[1][1].axis('off')
    axes[1][1].set_title('AI - WordCloud')

    plt.tight_layout()
    plt.savefig(f'{title.lower()}_words.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 저장 완료: {title.lower()}_words.png")

# 에세이 단어 시각화
visualize_words(df_essay, 'Essay')

# 뉴스 단어 시각화
visualize_words(df_news, 'News')

# 어려운 데이터 단어 시각화
visualize_words(df_hard, 'Hard')

# ------------------------------------------------------------------------------------------------------------------------------------------------ #




## 5. 전처리한 데이터로 고전분류모델 학습 (TF-IDF 기반)

### TF-IDF: 텍스트를 숫자로 변환하는 방법입니다.
#### 사람 글에는 이런 단어가 많고, AI 글에는 저런 단어가 많다를 숫자로 표현합니다.
##### 사용 모델 4가지
###### | Logistic Regression | 가장 기본적인 분류 모델 |
###### | Naive Bayes | 단어 확률 기반 분류 모델 |
###### | Random Forest | 여러 결정 트리를 합친 모델 |
###### | XGBoost | 강력한 앙상블 분류 모델 |

### 학습 방법
#### 전체 데이터의 80%(4,000개)로 학습
#### 나머지 20%(1,000개)로 성능 테스트


def run_models(df, domain_name):
    """TF-IDF + 고전 모델 4가지를 학습하고 성능을 출력합니다."""
    print(f"\n{'='*50}")
    print(f"  {domain_name} 데이터 모델 학습 시작")
    print(f"{'='*50}")

    # TF-IDF로 텍스트를 숫자로 변환
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X = tfidf.fit_transform(df['clean_text'])
    y = df['generated']

    # 80:20으로 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"학습용: {X_train.shape[0]}개 / 테스트용: {X_test.shape[0]}개\n")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes'        : MultinomialNB(),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost'            : XGBClassifier(eval_metric='logloss', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        results[name] = {'Accuracy': acc, 'F1-score': f1}
        print(f"[{name}]")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  F1-score : {f1*100:.2f}%")
        print()
    return results

# 에세이 고전 모델 학습
essay_results = run_models(df_essay, 'Essay')


# 뉴스 고전 모델 학습
news_results = run_models(df_news, 'News')


# 어려운 데이터 고전 모델 학습
hard_results = run_models(df_hard, 'Hard')

# ------------------------------------------------------------------------------------------------------------------------------------------------ #





## 6. 고전분류모델 성능 시각화

### 4가지 고전 모델의 정확도와 F1-score를 막대그래프로 비교
### Accuracy(정확도): 전체 중 맞게 예측한 비율
### F1-score: 정밀도와 재현율의 균형 점수 (불균형 데이터에서 더 신뢰할 수 있는 지표)


def visualize_classical(results, title, color):
    """고전 모델 성능을 시각화합니다."""
    keys = ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'XGBoost']
    model_names = ['Logistic\nRegression', 'Naive\nBayes', 'Random\nForest', 'XGBoost']
    accs = [results[k]['Accuracy'] * 100 for k in keys]
    f1s  = [results[k]['F1-score'] * 100 for k in keys]
    x = np.arange(len(model_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{title} - Classical Models Performance', fontsize=14, fontweight='bold')

    axes[0].bar(x, accs, width, color=color)
    axes[0].set_title('Accuracy (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].set_ylim(70, 102)
    axes[0].set_ylabel('Accuracy (%)')
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')

    axes[1].bar(x, f1s, width, color=color)
    axes[1].set_title('F1-score (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names)
    axes[1].set_ylim(70, 102)
    axes[1].set_ylabel('F1-score (%)')
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{title.lower()}_classical.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_classical(essay_results, 'Essay',       'steelblue')
visualize_classical(news_results,  'News',        'tomato')
visualize_classical(hard_results,  'Hard', 'gray')

# ------------------------------------------------------------------------------------------------------------------------------------------------ #


############### 최신 모델 학습하는 코드로 돌리면 30분 넘게 걸리십니다. 돌리지 마세요 !!! ###############
############### 간단하게 확인하고 싶으시면, 7-1번 코드 실행하세요 !!! ###############


## 7. 최신 모델 BERT 학습

### BERT란?
### 고전 모델들은 단어 빈도만 보지만, BERT는 문장 전체의 문맥과 의미를 이해한다.

### 사용 모델
#### DistilBERT: BERT의 경량화 버전 (속도 빠름, 성능 유사)

### 학습 방법
#### 전체 데이터의 80%(4,000개)로 학습
#### 나머지 20%(1,000개)로 성능 테스트


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class TextDataset(Dataset):
    """BERT 학습을 위해 텍스트를 토큰으로 변환하는 클래스입니다."""
    def __init__(self, texts, labels, max_len=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }

def run_bert(df, domain_name, epochs=3, sample_n=5000, max_len=128):
    """BERT 모델을 학습하고 성능을 출력합니다."""
    print(f"\n{'='*50}")
    print(f"  BERT - {domain_name} 학습 시작")
    print(f"{'='*50}")

    df = df.copy()
    df['generated'] = pd.to_numeric(df['generated'], errors='coerce')
    df = df.dropna(subset=['generated', 'text'])
    df['generated'] = df['generated'].astype(int)
    df = df[df['generated'].isin([0, 1])].reset_index(drop=True)

    # 샘플 수 조절
    n = min(sample_n // 2, len(df[df['generated']==0]), len(df[df['generated']==1]))
    df = df.groupby('generated').sample(n=n, random_state=42).reset_index(drop=True)
    print(f"사용 샘플: {len(df)}개 (사람: {n}개, AI: {n}개)")

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['generated'],
        test_size=0.2, random_state=42, stratify=df['generated']
    )
    print(f"학습용: {len(X_train)}개 / 테스트용: {len(X_test)}개")

    train_loader = DataLoader(TextDataset(X_train.values, y_train.values, max_len), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TextDataset(X_test.values,  y_test.values,  max_len), batch_size=32)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    print(f"\n✅ [BERT - {domain_name}]")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1-score : {f1*100:.2f}%")
    return {'Accuracy': acc, 'F1-score': f1}

# 에세이 BERT 학습 (약 10분 소요)
bert_essay = run_bert(df_essay, 'Essay', epochs=3, sample_n=5000)

# 뉴스 BERT 학습 (약 10분 소요)
bert_news = run_bert(df_news, 'News', epochs=3, sample_n=5000)

# 어려운 데이터 BERT 학습 (약 10분 소요)
bert_hard = run_bert(df_hard, 'Hard', epochs=3, sample_n=5000)

# ------------------------------------------------------------------------------------------------------------------------------------------------ #


############### 이거 돌리세요. 최신 모델 학습하는 방식으로 샘플 수를 줄였습니다. ###############


## 7-1. 최신 모델 BERT 학습 (샘플 수 5000 -> 1000 / 및 epoch 3->2 감소)

### BERT란?
### 고전 모델들은 단어 빈도만 보지만, BERT는 문장 전체의 문맥과 의미를 이해한다.

### 사용 모델
#### DistilBERT: BERT의 경량화 버전 (속도 빠름, 성능 유사)

### 학습 방법
#### 전체 데이터의 80%(4,000개)로 학습
#### 나머지 20%(1,000개)로 성능 테스트


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class TextDataset(Dataset):
    """BERT 학습을 위해 텍스트를 토큰으로 변환하는 클래스입니다."""
    def __init__(self, texts, labels, max_len=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }

def run_bert(df, domain_name, epochs=2, sample_n=1000, max_len=128):
    """BERT 모델을 학습하고 성능을 출력합니다."""
    print(f"\n{'='*50}")
    print(f"  BERT - {domain_name} 학습 시작")
    print(f"{'='*50}")

    df = df.copy()
    df['generated'] = pd.to_numeric(df['generated'], errors='coerce')
    df = df.dropna(subset=['generated', 'text'])
    df['generated'] = df['generated'].astype(int)
    df = df[df['generated'].isin([0, 1])].reset_index(drop=True)

    # 샘플 수 조절
    n = min(sample_n // 2, len(df[df['generated']==0]), len(df[df['generated']==1]))
    df = df.groupby('generated').sample(n=n, random_state=42).reset_index(drop=True)
    print(f"사용 샘플: {len(df)}개 (사람: {n}개, AI: {n}개)")

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['generated'],
        test_size=0.2, random_state=42, stratify=df['generated']
    )
    print(f"학습용: {len(X_train)}개 / 테스트용: {len(X_test)}개")

    train_loader = DataLoader(TextDataset(X_train.values, y_train.values, max_len), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TextDataset(X_test.values,  y_test.values,  max_len), batch_size=32)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    print(f"\n✅ [BERT - {domain_name}]")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1-score : {f1*100:.2f}%")
    return {'Accuracy': acc, 'F1-score': f1}


# 에세이 BERT 학습 (약 2분 소요)
bert_essay = run_bert(df_essay, 'Essay', epochs=2, sample_n=1000)

# 뉴스 BERT 학습 (약 2분 소요)
bert_news = run_bert(df_news, 'News', epochs=2, sample_n=1000)

# 어려운 데이터 BERT 학습 (약 2분 소요)
bert_hard = run_bert(df_hard, 'Hard', epochs=2, sample_n=1000)

# ------------------------------------------------------------------------------------------------------------------------------------------------ #



## 8. 고전모델 4개 vs 최신모델 BERT 최종결과 비교 시각화

### 세 가지 데이터셋에서 고전 모델 최고 성능과 BERT를 비교

### 핵심 포인트
#### 쉬운 데이터 (Essay/News): 두 모델 모두 높은 정확도
#### 어려운 데이터 (Hard): BERT가 고전 모델보다 확연히 높음
#### 데이터가 어려울수록 BERT의 강점이 드러난다


keys = ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'XGBoost']
all_models = ['Logistic\nRegression', 'Naive\nBayes', 'Random\nForest', 'XGBoost', 'BERT']
datasets = ['Essay', 'News', 'Hard']

# 각 데이터셋별 전체 모델 정확도
all_accs = {
    'Essay': [essay_results[k]['Accuracy']*100 for k in keys] + [bert_essay['Accuracy']*100],
    'News' : [news_results[k]['Accuracy']*100  for k in keys] + [bert_news['Accuracy']*100],
    'Hard' : [hard_results[k]['Accuracy']*100  for k in keys] + [bert_hard['Accuracy']*100],
}

x = np.arange(len(all_models))
width = 0.25
colors = ['steelblue', 'tomato', 'gray']

fig, ax = plt.subplots(figsize=(18, 7))
fig.suptitle('All Models Performance - Essay vs News vs Hard', fontsize=14, fontweight='bold')

for i, (ds, color) in enumerate(zip(datasets, colors)):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, all_accs[ds], width, label=ds, color=color, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', fontsize=7, fontweight='bold')

# BERT 영역 강조
ax.axvline(x=3.5, color='black', linestyle='--', alpha=0.4)
ax.text(3.6, 102, 'BERT →', fontsize=10, color='black')

ax.set_xticks(x)
ax.set_xticklabels(all_models)
ax.set_ylim(70, 105)
ax.set_ylabel('Accuracy (%)')
ax.legend(title='Dataset', loc='lower right')

plt.tight_layout()
plt.savefig('final_all_models.png', dpi=150, bbox_inches='tight')
plt.show()

# 결과 표
print("\n" + "="*70)
print(f"{'모델':<22} {'Essay':>10} {'News':>10} {'Hard':>10}")
print("-"*70)
for i, name in enumerate(['Logistic Regression', 'Naive Bayes', 'Random Forest', 'XGBoost', 'BERT']):
    print(f"{name:<22} {all_accs['Essay'][i]:>9.1f}% {all_accs['News'][i]:>9.1f}% {all_accs['Hard'][i]:>9.1f}%")
print("="*70)

# 최종 결과 CSV로 저장
keys = ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'XGBoost']
result_data = []
for name in keys:
    result_data.append({
        '모델': name,
        'Essay_Accuracy': round(essay_results[name]['Accuracy']*100, 2),
        'Essay_F1':       round(essay_results[name]['F1-score']*100, 2),
        'News_Accuracy':  round(news_results[name]['Accuracy']*100, 2),
        'News_F1':        round(news_results[name]['F1-score']*100, 2),
        'Hard_Accuracy':  round(hard_results[name]['Accuracy']*100, 2),
        'Hard_F1':        round(hard_results[name]['F1-score']*100, 2),
    })
result_data.append({
    '모델': 'BERT',
    'Essay_Accuracy': round(bert_essay['Accuracy']*100, 2),
    'Essay_F1':       round(bert_essay['F1-score']*100, 2),
    'News_Accuracy':  round(bert_news['Accuracy']*100, 2),
    'News_F1':        round(bert_news['F1-score']*100, 2),
    'Hard_Accuracy':  round(bert_hard['Accuracy']*100, 2),
    'Hard_F1':        round(bert_hard['F1-score']*100, 2),
})

df_results = pd.DataFrame(result_data)
df_results.to_csv('final_results.csv', index=False, encoding='utf-8-sig')
print("✅ 최종 결과 CSV 저장 완료: final_results.csv")
print("✅ 시각화 이미지 저장 완료: final_all_models.png")
print()
print(df_results.to_string(index=False))

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

# ======================================================================
# 모델                          Essay       News       Hard
# ----------------------------------------------------------------------
# Logistic Regression         97.6%      99.1%      87.6%
# Naive Bayes                 95.9%      93.7%      82.2%
# Random Forest               97.7%      97.7%      86.9%
# XGBoost                     97.8%      99.2%      85.9%
# BERT                        97.7%      99.2%      93.3%
# ======================================================================

