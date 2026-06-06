# Processed Data

이 폴더는 본 프로젝트에서 전처리를 완료한 데이터를 저장하는 공간이다.  
원본 데이터는 `data/project_dataset/` 폴더에 위치하며, 메인 코드인 `notebooks/human_vs_ai_text_classification.py`를 실행하면 전처리된 데이터가 이 폴더에 생성된다.

본 프로젝트에서는 Essay, News, Hard dataset을 사용하였다.  
Essay와 News는 main experiment에 사용되었고, Hard dataset은 additional experiment에 사용되었다.

---

## 0. Datasets

본 프로젝트에서는 세 가지 데이터셋을 사용하였다.
|Dataset|Human-written|Ai-generated|Total|Usage|
|---|---|---|---|---|
|Essay|2500|2500|5000|Main Experiment|
|News|2500|2500|5000|Main Experiment|
|Hard| - | - | - |Additional Experiment|

Essay와 News는 장르별 Ai탐지 성능을 확인하기 위한 데이터셋이다
Hard는 ChatGPT가 생성한 텍스트를 포함하고 있어, 더 어려운 조건에서 모델 성능을 확인하기 위한 추가 실험 데이터셋으로 활용하였다.
ChatGPT는 문맥을 고려하여 자연스러운 문장을 생성하기 때문에, 단순한 단어 빈도 기반 모델로는 구분하기 어렵다.
이 때문에 고전 ML모델의 성능이 낮게 나타날 수 있다.

Essay와 News는 문체의 특성이 서로 다르기 때문에 하나로 합치지 않고 각각 독립적인 분류 문제로 설계하였다.
Essay는 개인의 경험, 주장, 감정 표현이 많이 나타나는 반면, News는 객관적이고 형식적인 문장 구조를 가지는 경우가 많다.
Essay와 News데이터를 합쳐서 학습시킬 경우, 모델이 Ai여부가 아닌 장르차이를 학습할 가능성이 있어 이를 방지하였다.

---
## 1. Purpose of Processed Data

원본 텍스트 데이터는 그대로 모델 학습에 사용하기 어렵다.  
텍스트 안에는 대소문자 차이, 특수문자, 불필요한 공통 단어, 깨진 문자 등이 포함되어 있을 수 있기 때문이다.

따라서 본 프로젝트에서는 원본 텍스트를 모델이 학습하기 쉬운 형태로 정리한 뒤, 전처리된 데이터를 별도로 저장하였다.
전처리 전후 예시는 다음과 같다

`Before` "Furthermore, it's worth nothing that students' running habits - café visits & social media usage..."

`After`  "furthermore worth note student run habit visit social medium use"

전처리를 통해서 모델이 불필요한 기호나 형태 차이에 흔들리지 않고, 텍스트의 핵심적인 단어 패턴을 더 잘 학습할 수 있도록 하였다.
전처리된 데이터는 모델 학습과 시각화 분석에 사용된다.

---

## 2. Generated Files

코드를 실행하면 다음 파일들이 생성된다.

```text
essay_clean_dataset.csv
news_clean_dataset.csv
hard_clean_dataset.csv
```

각 파일의 의미는 다음과 같다.

| File | Description | Usage |
|---|---|---|
| essay_clean_dataset.csv | Essay dataset을 전처리한 데이터 | Main Experiment |
| news_clean_dataset.csv | News dataset을 전처리한 데이터 | Main Experiment |
| hard_clean_dataset.csv | Hard dataset을 전처리한 데이터 | Additional Experiment |

---

## 3. Data Columns

전처리된 데이터는 기본적으로 다음과 같은 컬럼을 포함한다.

| Column | Description |
|---|---|
| text | 원본 텍스트 |
| generated | AI 생성 여부를 나타내는 label |
| clean_text | 전처리를 거친 텍스트 |

### Label 정의
Label의 의미는 다음과 같다.

| Label | Meaning |
|---|---|
| 0 | Human-written text |
| 1 | AI-generated text |

`text` 컬럼은 원본 문장을 보존하기 위한 컬럼이고, `clean_text` 컬럼은 실제 모델 학습에 사용되는 전처리 결과이다.
label은 'generated' 컬럼에 저장되어 있으며, 전처리 과정에서 0과 1 이외의 값은 제거한다.

---

## 4. Preprocessing Steps

전처리 과정은 메인 코드의 `clean_text()` 함수에서 수행된다.

주요 전처리 과정은 다음과 같다.

```text
1. 이상기호 일반 문자 치환
2. 영어 외 문자 제거
3. 소문자 변환
4. 알파벳과 공백만 남기기
5. 불용어 제거
6. Lemmatization
7. 짧은 텍스트 제거
```
전처리과정이 필요한 이유는 텍스트 데이터를 모델에 바로 넣을 수 없기 때문이다. 텍스트에는 대문자, 특수문자, 의미 없는 단어들이 섞여있으며, 그대로 사용할 경우 모델이 불필요한 패턴을 학습할 수 있다. 즉 모델이 텍스트의 핵심적인 단어 패턴을 잘 학습하도록 하기 위한 과정이다.
전처리 단계 요약
| 단계 | 내용 | 적용 이유 |
|-----|------|---------|
| 1 | 이상기호 일반 문자 변환 | 따옴표, 대시 등 특수 문자가 모델 오류를 유발할 수 있어 일반 문자로 변환 |
| 2 | 영어 외 문자 제거 | 한글, 이모지 등 영어 텍스트 분석에 불필요한 문자 제거 |
| 3 | 소문자 변환 | "The"와 "the"를 같은 단어로 처리하기 위해 |
| 4 | 알파벳과 공백만 남기기 | 숫자, 기호 등 텍스트 분류와 무관한 문자 제거 |
| 5 | 불용어 제거 | "the", "is" 등 분류에 도움이 되지 않는 단어 제거 |
| 6 | Lemmatization | "running"→"run"으로 통일해 중복 카운트 방지 |
| 7 | 짧은 텍스트 제거 | 글자 수가 너무 적으면 패턴 학습이 불가능하므로 제거 |

---

## 5. Why These Preprocessing Steps?
### 5.0 깨진 문자 치환
워드나 웹에서 복붙한 텍스트에는 유니코드 특수문자가 섞여 있는 경우가 많다.
이러한 문자를 그대로 두면 같은 의미의 문장이 서로 다른 토큰으로 처리될 수 있다.
| 원래 문자 | 의미 | 변환 결과 |
|---------|------|---------|
| `\u2018`, `\u2019` | 예쁜 따옴표 | `'` (일반 따옴표) |
| `\u201C`, `\u201D` | 예쁜 쌍따옴표 | `"` (일반 쌍따옴표) |
| `\u2014`, `\u2013` | 긴 대시 | `-` (하이픈) |
| `\u00A0` | 줄바꿈 없는 공백 | ` ` (일반 공백) |
| `\u2026` | 말줄임표 | `...` | 

따라서 이러한 문자들을 일반 ASCII 문자로 통일하여
같은 의미의 표현이 다르게 처리되는 문제를 방지하였다.
### 5.1 소문자 변환

영어 텍스트에서는 같은 단어라도 대문자와 소문자가 섞여 있으면 서로 다른 단어로 처리될 수 있다.

예를 들어 `AI`, `Ai`, `ai`는 의미상 같은 단어이지만, 전처리하지 않으면 서로 다른 토큰처럼 인식될 수 있다.  
따라서 모든 텍스트를 소문자로 변환하여 단어 표현을 통일하였다.

---

### 5.2 특수문자 제거

텍스트에는 쉼표, 따옴표, 괄호, 특수기호 등이 포함되어 있을 수 있다.  
이러한 문자는 일부 문맥에서는 의미를 가질 수 있지만, 본 프로젝트의 TF-IDF 기반 분류에서는 단어 사용 패턴을 중심으로 분석하기 때문에 제거하였다.

이를 통해 모델이 불필요한 기호보다 실제 단어에 집중할 수 있도록 하였다.
e.g. hello! I'm a student. -> hello im a student
e.g. café → caf

---

### 5.3 불용어 제거

`the`, `is`, `and`, `of`와 같은 단어는 대부분의 영어 문서에서 매우 자주 등장한다.  
그러나 이러한 단어는 AI-generated text와 human-written text를 구분하는 데 유용한 정보를 제공하지 않는다.

따라서 불용어를 제거하여 분류에 더 도움이 될 수 있는 단어 중심으로 분석하였다.
e.g. I am going to the school -> going school

본 프로젝트에서는 영어 불용어 목록파일을 이용하였다. 직접 만든 것이 아닌 NLTK가 사전에 정의한 목록을 그대로 활용하였다

---

### 5.4 Lemmatization

Lemmatization은 단어를 원형에 가깝게 바꾸는 과정이다(어간 추출).

예를 들어 다음과 같은 단어들은 형태는 다르지만 비슷한 의미를 가진다.

```text
running → run
studies → study
better → good
```

이러한 단어들을 정리하면, 모델이 같은 의미의 단어를 너무 여러 개의 다른 단어로 나누어 학습하는 문제를 줄일 수 있다.

---

### 5.5 짧은 텍스트 제거

너무 짧은 텍스트는 AI 글과 사람 글의 특징을 충분히 담고 있지 않을 수 있다.  
예를 들어 단어가 몇 개밖에 없는 문장은 문체, 단어 선택, 문장 구조의 차이를 파악하기 어렵기 때문에 모델 학습에 오히려 노이즈가 될 수 있다.

따라서 본 프로젝트에서는 전처리 후 단어 수가 너무 적은 텍스트를 제거하였다.  
이를 통해 모델이 충분한 정보를 가진 텍스트를 기준으로 학습하도록 하였다.
e.g. hello -> 제거
e.g. hello run school learn study -> 유지

### 5.6 TF-IDF와 전처리의 관계
본 프로젝트에서 텍스트를 숫자로 변환하는데 사용한 방법은 TF-IDF이다.

TF-IDF는 다음 두 가지를 곱한 값이다.
- **TF (Term Frequency)**: 이 문서에서 이 단어가 얼마나 자주 나오는가
- **IDF (Inverse Document Frequency)**: 이 단어가 전체 문서에서 얼마나 희귀한가
예를 들어 'furthermore'라는 단어가 Ai글에서만 자주 나오고 사람 글에서는 잘 나오지 않는다면, 이 단어는 높은 TF-IDF 점수를 받고 Ai글 판별에 중요한 단어로 학습된다.

---
전처리를 통해 불용어, 특수문자, 중복 형태의 단어를 미리 제거하면 TF-IDF가 실제로 의미 있는 단어에 더 정확한 점수를 부여할 수 있다.
결과적으로 TF-IDF를 입력으로 받는 고전 ML 모델의 분류 성능도 함께 향상된다.
즉, 전처리는 TF-IDF와 고전 ML 모델 전체의 성능을 높이기 위한 필수 과정이다.

#### 추가로 할 수 있는 전처리
##### 1. 철자교정
`studnet → student`

`teh → the`
   
사람 글에는 오타가 많고 Ai글에는 거의 없다. 만약 오타가 존재하는 글에 대해 철자 교정 전처리를 적용한다면, Ai의 판단기준이 하나 사라지는 셈이 되니, 오히려 이 전처리는 안하는 것이 나을 수도 있다.

##### 2. n-gram범위 확장
   `현재: ngram_range=(1,2) → 단어 1~2개 묶음`
   
   `확장: ngram_range=(1,3) → 단어 1~3개 묶음`

   Ai글에서 자주 나오는 3단어 표현까지 학습할 수 있게되어, 성능이 좋아질 수도 있다.

##### 3. 구두점 패턴 보존
   현재: 구두점 전부 제거
   개선: AI 글 특유의 구두점 패턴을 피처로 활용

AI 글은 세미콜론(;), 콜론(:) 을 사람보다 많이 쓴다. 제거하지 않고 패턴으로 학습하면 성능 향상 가능

##### 4. 문장 길이를 판단기준에 추가
Ai글과 사람 글의 문장 길이에 대한 정보를 학습하여 TF-IDF와 함께 입력하면 성능 향상이 가능

---

## 6. File Size Limitation

GitHub 웹 업로드에는 파일 크기 제한이 있다.  
본 프로젝트에서 전처리된 Essay dataset은 이 폴더에 업로드하였으나, `news_clean_dataset.csv` 파일은 용량 제한을 초과하여 업로드하지 않았다.

따라서 `news_clean_dataset.csv`는 GitHub 저장소에는 포함되어 있지 않지만, 프로젝트 실행에는 문제가 없다.  
메인 코드의 전처리 과정을 실행하면 동일한 파일을 로컬 환경에서 다시 생성할 수 있다.

생성되는 파일은 다음과 같다.

```text
news_clean_dataset.csv

해당 파일은 다음 명령어를 실행하면 생성된다.

python notebooks/human_vs_ai_text_classification.py

즉, news_clean_dataset.csv는 저장소에 직접 업로드된 파일이 아니라, 코드 실행을 통해 재현 가능한 전처리 결과물이다.

---

## 7. Reproducing Processed Data

전처리 데이터를 다시 생성하려면 프로젝트 최상위 폴더에서 다음 명령어를 실행하면 된다.

```bash
python notebooks/human_vs_ai_text_classification.py
```

코드가 실행되면 다음 경로에 전처리된 데이터가 저장된다.

```text
data/processed/
├── essay_clean_dataset.csv
├── news_clean_dataset.csv
└── hard_clean_dataset.csv
```

---

## 8. Role in the Project

이 폴더의 데이터는 단순한 중간 결과물이 아니라, 모델 학습과 결과 해석의 기반이 된다.

전처리된 데이터는 다음 단계에서 사용된다.

```text
1. Word frequency analysis
2. WordCloud visualization
3. TF-IDF vectorization
4. Classical machine learning model training
5. BERT-based model training
6. Final performance comparison
```

즉, processed data는 원본 데이터와 모델링 결과를 연결하는 핵심 단계라고 볼 수 있다.

---

## 9. Notes

- 이 폴더의 파일은 메인 코드를 실행하면 다시 생성할 수 있다.
- 용량 문제로 일부 파일이 GitHub에 없더라도, 원본 데이터와 코드가 있으면 재현 가능하다.
- `clean_text` 컬럼은 실제 모델 학습에 사용되는 핵심 컬럼이다.
- Essay와 News는 main experiment, Hard는 additional experiment로 사용된다.
- 본 전처리 과정은 단순 자동화가 아니라, 모델이 텍스트 패턴을 더 명확하게 학습하도록 하기 위한 판단에 기반한다.
