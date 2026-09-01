import re
SKILLS=['python','java','c++','c#','sql','machine learning','deep learning','nlp','tensorflow','pytorch','pandas','numpy','scikit-learn','excel','power bi','tableau','aws','azure','docker','kubernetes','git','linux','html','css','javascript','react','node.js','django','flask','spring boot','mongodb','mysql','postgresql','spark','hadoop','communication','leadership','project management','recruitment','marketing','autocad','solidworks']
def extract_skills(text):
    t=str(text).lower(); return sorted({s for s in SKILLS if re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)',t)})
def skill_gap(resume_text, job_text):
    r=set(extract_skills(resume_text)); j=set(extract_skills(job_text)); return sorted(r),sorted(j-r)
