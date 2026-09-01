import os, pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out=os.path.join(ROOT,'data','processed','jobs_clean.csv')
os.makedirs(os.path.dirname(out),exist_ok=True)
rows=[
('Python Developer','Demo Tech','Bengaluru','Python, Django, Flask, SQL, Git','Build Python applications, REST APIs and data-processing services.','1-3 years'),
('Data Scientist','Demo Analytics','Bengaluru','Python, Machine Learning, Pandas, NumPy, Scikit-learn, SQL','Develop machine learning models, analyze data and evaluate predictive performance.','1-3 years'),
('Data Analyst','Demo Insights','Bengaluru','SQL, Excel, Python, Power BI, Statistics','Analyze business data, create dashboards and communicate insights.','0-2 years'),
('Java Developer','Demo Software','Hyderabad','Java, Spring Boot, SQL, REST API, Git','Develop enterprise Java applications and backend services.','1-3 years'),
('Web Developer','Demo Web','Bengaluru','HTML, CSS, JavaScript, React, Git','Create responsive web applications and maintain frontend components.','0-2 years'),
('DevOps Engineer','Demo Cloud','Pune','Linux, Docker, Kubernetes, AWS, CI/CD, Git','Automate deployment pipelines and maintain cloud infrastructure.','2-4 years'),
('HR Executive','Demo People','Bengaluru','Recruitment, HR, Communication, Excel','Support recruitment, employee coordination and HR operations.','0-2 years'),
('Mechanical Engineer','Demo Engineering','Chennai','AutoCAD, SolidWorks, CAD, Manufacturing','Design mechanical components and support manufacturing processes.','1-3 years'),
('Civil Engineer','Demo Infra','Bengaluru','AutoCAD, Civil Engineering, Construction, Project Management','Support construction planning, drawings and site execution.','1-3 years'),
('Marketing Specialist','Demo Brand','Bengaluru','Marketing, SEO, Social Media, Analytics, Content','Plan digital campaigns and measure marketing performance.','0-3 years')]
df=pd.DataFrame(rows,columns=['title','company','location','skills','description','experience'])
df['text']=(df['title']+' '+df['skills']+' '+df['description']+' '+df['experience']).str.lower()
df.to_csv(out,index=False)
print('Created demo job corpus:',out)
print('IMPORTANT: replace this demo file with a real public Naukri/LinkedIn job corpus for the final submission.')
