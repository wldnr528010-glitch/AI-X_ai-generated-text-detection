# Processed Data

이 폴더는 본 프로젝트에서 전처리를 완료한 데이터를 저장하는 공간이다.  
원본 데이터는 `data/project_dataset/` 폴더에 위치하며, 메인 코드인 `notebooks/human_vs_ai_text_classification.py`를 실행하면 전처리된 데이터가 이 폴더에 생성된다.

본 프로젝트에서는 Essay, News, Hard dataset을 사용하였다.  
Essay와 News는 main experiment에 사용되었고, Hard dataset은 additional experiment에 사용되었다.

---

## 1. Purpose of Processed Data

원본 텍스트 데이터는 그대로 모델 학습에 사용하기 어렵다.  
텍스트 안에는 대소문자 차이, 특수문자, 불필요한 공통 단어, 깨진 문자 등이 포함되어 있을 수 있기 때문이다.

따라서 본 프로젝트에서는 원본 텍스트를 모델이 학습하기 쉬운 형태로 정리한 뒤, 전처리된 데이터를 별도로 저장하였다.

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

Label의 의미는 다음과 같다.

| Label | Meaning |
|---|---|
| 0 | Human-written text |
| 1 | AI-generated text |

`text` 컬럼은 원본 문장을 보존하기 위한 컬럼이고, `clean_text` 컬럼은 실제 모델 학습에 사용되는 전처리 결과이다.

---

## 4. Preprocessing Steps

전처리 과정은 메인 코드의 `clean_text()` 함수에서 수행된다.

주요 전처리 과정은 다음과 같다.

```text
1. 깨진 문자 치환
2. 소문자 변환
3. ASCII 범위 밖 문자 제거
4. 알파벳과 공백만 남기기
5. 불용어 제거
6. Lemmatization
7. 너무 짧은 텍스트 제거
```

각 단계는 단순히 데이터를 줄이기 위한 것이 아니라, 모델이 텍스트의 핵심적인 단어 패턴을 더 잘 학습하도록 하기 위한 과정이다.

---

## 5. Why These Preprocessing Steps?

### 5.1 소문자 변환

영어 텍스트에서는 같은 단어라도 대문자와 소문자가 섞여 있으면 서로 다른 단어로 처리될 수 있다.

예를 들어 `AI`, `Ai`, `ai`는 의미상 같은 단어이지만, 전처리하지 않으면 서로 다른 토큰처럼 인식될 수 있다.  
따라서 모든 텍스트를 소문자로 변환하여 단어 표현을 통일하였다.

---

### 5.2 특수문자 제거

텍스트에는 쉼표, 따옴표, 괄호, 특수기호 등이 포함되어 있을 수 있다.  
이러한 문자는 일부 문맥에서는 의미를 가질 수 있지만, 본 프로젝트의 TF-IDF 기반 분류에서는 단어 사용 패턴을 중심으로 분석하기 때문에 제거하였다.

이를 통해 모델이 불필요한 기호보다 실제 단어에 집중할 수 있도록 하였다.

---

### 5.3 불용어 제거

`the`, `is`, `and`, `of`와 같은 단어는 대부분의 영어 문서에서 매우 자주 등장한다.  
그러나 이러한 단어는 AI-generated text와 human-written text를 구분하는 데 큰 정보를 제공하지 않을 수 있다.

따라서 불용어를 제거하여 분류에 더 도움이 될 수 있는 단어 중심으로 분석하였다.

---

### 5.4 Lemmatization

Lemmatization은 단어를 원형에 가깝게 바꾸는 과정이다.

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
예를 들어 단어가 몇 개밖에 없는 문장은 문체, 단어 선택, 문장 구조의 차이를 파악하기 어렵다.

따라서 본 프로젝트에서는 전처리 후 단어 수가 너무 적은 텍스트를 제거하였다.  
이를 통해 모델이 충분한 정보를 가진 텍스트를 기준으로 학습하도록 하였다.

---

## 6. Why Processed Data May Not Be Fully Uploaded

GitHub 웹 업로드에는 파일 크기 제한이 있다.  
따라서 일부 전처리된 데이터 파일, 특히 용량이 큰 파일은 저장소에 직접 업로드되지 않을 수 있다.

예를 들어 `news_clean_dataset.csv` 또는 기타 전처리 파일이 용량 제한을 초과할 경우, 해당 파일은 GitHub에 업로드하지 않고 로컬 환경에서 생성하여 사용한다.

이 경우에도 프로젝트 실행에는 문제가 없다.  
메인 코드를 실행하면 동일한 전처리 과정이 수행되고, 이 폴더에 clean dataset이 다시 생성된다.

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
