from sentence_transformers import SentenceTransformer, util
import json, random
from data_augmentation import CGDAugmentor
kr = SentenceTransformer("nlp/app_project/kr_first_tuned")

with open("nlp/app_project/CourseAdvisor-Bot/course_advisor.courses.json", "r") as f:
    rows = json.load(f)

aug = CGDAugmentor()

course = random.choice(rows)
doc    = aug.format_course_info(course, "course_info")
query  = f"Tell me about {course['code']}"

print(util.cos_sim(
        kr.encode(query, convert_to_tensor=True),
        kr.encode(doc ,  convert_to_tensor=True)))
