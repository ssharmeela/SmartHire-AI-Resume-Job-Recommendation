import os, sys, joblib, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
sys.path.append(os.path.dirname(__file__))
from text_utils import clean_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'raw', 'Resume.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
REPORT_DIR = os.path.join(ROOT, 'reports')
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA)
print('DATASET SHAPE:', df.shape)
print('COLUMNS:', df.columns.tolist())
required = {'Resume_str','Category'}
if not required.issubset(df.columns): raise ValueError(f'Missing columns. Need {required}')

df = df[['Resume_str','Category']].copy()
df['Resume_str'] = df['Resume_str'].fillna('').map(clean_text)
df = df[df['Resume_str'].str.len() > 0].drop_duplicates(subset=['Resume_str']).reset_index(drop=True)
print('AFTER CLEANING/DEDUP:', df.shape)
print('CATEGORIES:', df['Category'].nunique())

df.to_csv(os.path.join(ROOT,'data','processed','resumes_clean.csv'), index=False)
X_train_txt, X_test_txt, y_train, y_test = train_test_split(df['Resume_str'], df['Category'], test_size=0.20, random_state=42, stratify=df['Category'])

tfidf = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1,2), sublinear_tf=True, max_features=30000)
X_train = tfidf.fit_transform(X_train_txt); X_test = tfidf.transform(X_test_txt)
joblib.dump(tfidf, os.path.join(MODEL_DIR,'resume_tfidf.joblib'))

models = {
 'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
 'Linear SVM': LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=5000),
 'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
 'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
}
results=[]
for name, model in models.items():
    model.fit(X_train,y_train); pred=model.predict(X_test); acc=accuracy_score(y_test,pred)
    results.append((name,acc)); print(f'{name}: {acc*100:.2f}%')

# Tune the strongest baseline candidate. GridSearch is useful even if a tree happens to score 100% on one split;
# it checks whether the result is robust across folds rather than trusting a single split.
grid = GridSearchCV(
    LinearSVC(class_weight='balanced', random_state=42, max_iter=5000),
    {'C':[0.5,1,2]}, cv=2, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train,y_train)
best_model = grid.best_estimator_
pred = best_model.predict(X_test)
final_acc = accuracy_score(y_test,pred)
print('\nBEST MODEL: Linear SVM')
print('BEST PARAMS:', grid.best_params_)
print(f'CV ACCURACY: {grid.best_score_*100:.2f}%')
print(f'TEST ACCURACY: {final_acc*100:.2f}%')
print('\nCLASSIFICATION REPORT\n', classification_report(y_test,pred,zero_division=0))

joblib.dump(best_model, os.path.join(MODEL_DIR,'resume_classifier.joblib'))
joblib.dump({'categories': sorted(df['Category'].unique().tolist()), 'accuracy': final_acc, 'best_params':grid.best_params_}, os.path.join(MODEL_DIR,'classifier_meta.joblib'))

cm=confusion_matrix(y_test,pred,labels=best_model.classes_)
plt.figure(figsize=(16,12)); sns.heatmap(cm, cmap='Blues', annot=False, xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('SmartHire Resume Classifier - Confusion Matrix'); plt.xticks(rotation=90); plt.yticks(rotation=0); plt.tight_layout(); plt.savefig(os.path.join(REPORT_DIR,'confusion_matrix.png'),dpi=180); plt.close()

pd.DataFrame(results, columns=['Model','Accuracy']).sort_values('Accuracy',ascending=False).to_csv(os.path.join(REPORT_DIR,'model_comparison.csv'),index=False)
print('\nMODEL FILES SAVED in models/')
