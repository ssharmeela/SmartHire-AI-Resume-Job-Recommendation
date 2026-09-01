import os, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS=os.path.join(ROOT,'data','processed','jobs_clean.csv')
MODEL=os.path.join(ROOT,'models','job_tfidf.joblib')

def train_job_index():
    jobs=pd.read_csv(JOBS).fillna('')
    vec=TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=1)
    X=vec.fit_transform(jobs['text'].astype(str))
    joblib.dump(vec,MODEL)
    return jobs,vec,X

def recommend(resume_text, top_n=5):
    jobs,vec,X=train_job_index()
    q=vec.transform([resume_text])
    scores=cosine_similarity(q,X).ravel()
    idx=scores.argsort()[::-1][:top_n]
    out=jobs.iloc[idx][['title','company','location','skills','experience']].copy()
    out['match_score']=(scores[idx]*100).round(2)
    return out.reset_index(drop=True)
