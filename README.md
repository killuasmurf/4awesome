# Problem Statement

**Design and implement an ML-based system to evaluate the quality and relevancy of Google location reviews. The system should:**

- Gauge review quality: Detect spam, advertisements, irrelevant content, and rants from users who have likely never visited the location.
- Assess relevancy: Determine whether the content of a review is genuinely related to the location being reviewed.
- Enforce policies: Automatically flag or filter out reviews that violate the following example policies:
  - No advertisements or promotional content.
  - No irrelevant content (e.g., reviews about unrelated topics).
  - No rants or complaints from users who have not visited the place (can be inferred from content, metadata, or other signals).

## Data Source
Google Review Data: Open datasets containing Google location reviews (e.g., Google Local Reviews on Kaggle: https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews)

# Steps & Task Split

## How to Navigate Through Our ML Pipeline
In the [Google Drive folder](https://drive.google.com/drive/folders/1DiM0XhLP2lDE6daKbU0Pko-RVH0KjDrA?usp=drive_link), we have a step 0 file to load the data, and the following 6 steps in separate notebooks to carry out different functions.

---

## 1) Data Preprocessing & Cleaning
### 1. Basic Cleaning
- `drop_duplicates()` : removes duplicate rows.  
- `dropna(subset=["text"])` : removes rows without review text.  
- Prints dataset shape after cleaning.

### 2. Text Cleaning Function (`clean_text`)
Also a feature engineering step for the `cleaned_text` column:
- Converts text to lowercase.
- Removes:
  - URLs (`http...`, `www...`)
  - Extra spaces
  - Email addresses
  - Phone numbers (e.g., `123-456-7890` or `1234567890`)
  - User mentions (`@username`)
- Tokenizes into words.
- Removes English stopwords (`the`, `is`, `at`).
- Keeps only English reviews.
- Applies lemmatization (`running → run`).
- Joins cleaned tokens back into a string.
- Saves result as a new column `cleaned_text`.

---

## 2) EDA (Adrian)
### 1. Dataset Overview
- Displays dataset info, missing values, and summary statistics.
- Helps validate data quality before deeper analysis.

### 2. Target Variable Analysis: `rating_category`
- Distribution of target variable (`rating_category`) in raw counts & percentages.
- Visualized with bar charts & histograms.
- Heatmap of ratings × categories for consistency checks.
- Compares average rating per category.
- Analyzes review text length distribution (histogram + boxplot).
- Provides descriptive stats of text length per category.

### 3. Keyword and Topic Analysis
- **Top Words by Category**: frequency counts.
- **TF-IDF Distinctive Words**: highlights words distinctive to each category.

### 4. Spam / Advertising Detection
Rule-based scoring (`check_spam_content`):
- Spam score assigned based on:
  - Promotional keywords
  - URLs, emails, phone numbers
  - Repetitive words (≥3 times)
  - Too short or long reviews
  - Excessive punctuation or ALL CAPS
- Labels:
  - Genuine (≤1)
  - Suspicious (=2)
  - Likely Spam (≥3)
- New columns: `spam_score`, `spam_label`.

EDA Visualizations:
- Distribution of spam labels.
- Spam % by rating.
- Spam % by category.
- Scatterplot: Review length vs Spam score.

### 5. Sentiment Analysis
- Validates review authenticity & consistency.  
- Uses **TextBlob** to compute polarity (−1 to +1) and subjectivity (0 to 1).
- Stores in: `sentiment_polarity`, `sentiment_subjectivity`.
- Summary stats (overall averages).
- Category-level insights (mean polarity & subjectivity per aspect).
- Correlation between rating & polarity.
- Visualizations:
  - Polarity distribution
  - Scatterplot polarity vs rating
  - Average polarity by category
  - Boxplots of polarity/subjectivity by category

### 6. Correlation Analysis
- Investigates relationships between numeric features:
  - `rating`, `text_length`, `cleaned_text_length`, `spam_score`,
    `sentiment_polarity`, `sentiment_subjectivity`.
- Uses Pearson correlation.
- Heatmap with seaborn.
- Extracts correlations with `rating`.

---

## 3) Feature Engineering (Adrian + Kaixin)
- **Average Word Length**: indicates review quality.  
- **Unique Word Ratio**: detects spam/repetition.  
- **Sentiment Features (VADER)**:
  - `vader_pos`, `vader_neg`, `vader_neu`, `vader_compound`.

---

## 4) Policy Enforcement (Xian Rui)
1. **No Ads Policy** → detects promotional content.  
2. **Minimum Effort Policy**:
   - `cleaned_text_length < 5` → too short
   - `unique_word_ratio < 0.3` → repetitive/spammy
3. **Rating-Sentiment Consistency Policy**:
   - Compares `rating` vs `vader_compound` for mismatches.

---

## 5.1) Logistic Regression Model (Xian Rui)
### Target Variables
- `rating` (1–5) → customer satisfaction.
- `rating_category` → review aspect (e.g., taste, service).

### Process
- Train/test split.
- TF-IDF vectorization (unigrams + bigrams, top 5000 terms).
- Logistic Regression for:
  - **Aspect classification**
  - **Sentiment classification**
- Evaluation:
  - Accuracy, precision, recall, F1
  - Confusion matrix
  - SHAP for interpretability

---

## 5.2) Multi-Task BERT Transformer Model (Xian Rui)
### Setup
- Dataset: `reviews_with_policy_flags.csv`
- Splits: 80/20
- Features: `cleaned_text`
- Labels: rating, rating_category_encoded, policy flags

### Training
- Tokenization with BERT tokenizer.
- Multi-head architecture:
  - Rating prediction
  - Rating category
  - Policy ads
  - Policy short
  - Policy mismatch
- Loss: summed across tasks
- Optimizer: Adam (1e-5)
- Epochs: 5

### Results
- Rating accuracy: ~46%
- Category accuracy: ~35%
- Policy detection:
  - `policy_short`: good (F1 ~0.90)
  - `policy_ads`, `policy_mismatch`: poor due to class imbalance.

---

## 5.3) Large Language Model (Lin Myat & Darius): Prompt Engineering
- Dataset: `reviews_with_features.csv`
- Few-shot prompting with GPT-4o-mini API.

### Key Outputs
- **is_ad** → promotional content  
- **did_not_visit** → reviewer didn’t visit  
- **relevant_to_restaurant** → relevance scale  
- **evidence_snippets** → justification  

### Testing
- Used a test pool of edge cases (ads, non-visits, irrelevant, genuine).  
- Results stored in DataFrame.  

### Integration
- Merged LLM outputs with policy flags.
- Handled nulls with defaults (`did_not_visit = True`, irrelevant = "Very irrelevant").
- Created `policy_irrelevant` flag.
- Computed **Policy Violation Percentage**:
  - Based on: `policy_ads`, `policy_short`, `policy_mismatch`, `policy_novisit`, `policy_irrelevant`.
  - Converted bool → int, summed, divided by total
- Final dataset is stored as `reviews_final.csv`
 
<img width="543" height="519" alt="image" src="https://github.com/user-attachments/assets/e906c2eb-0db0-42a0-b4af-b1114c5cf3f2" />



