import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(ROOT,'src'))
print('STEP 1/4: creating demo job corpus')
import build_demo_jobs; build_demo_jobs.main() if hasattr(build_demo_jobs,'main') else None
print('STEP 2/4: training resume classifier')
import train_classifier
print('STEP 3/4: clustering jobs')
import clustering; clustering.run()
print('STEP 4/4: training job TF-IDF index')
import recommender; recommender.train_job_index()
print('DONE. Start Streamlit with: streamlit run app/streamlit_app.py')
