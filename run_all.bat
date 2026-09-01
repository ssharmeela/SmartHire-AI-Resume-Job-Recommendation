@echo off
call .venv\Scripts\activate
python src\build_demo_jobs.py
python src\train_classifier.py
python -c "import sys; sys.path.insert(0,'src'); import clustering; clustering.run()"
streamlit run app\streamlit_app.py
