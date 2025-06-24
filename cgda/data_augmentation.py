import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Tuple
import random
import numpy as np
from tqdm import tqdm

class CGDAugmentor:
    def __init__(self, model_name: str = "t5-base"):
        """
        Initialize the CGDA augmentor.
        
        Args:
            model_name (str): Name of the pre-trained model to use for augmentation
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Question templates for different types of queries
        self.question_templates = {
            "prerequisites": [
                "What are the prerequisites for {course_name} ({course_code})?",
                "Can you tell me what courses I need to take before enrolling in {course_name}?",
                "What are the required courses before taking {course_code}?",
                "Do I need any prerequisites for {course_name} ({course_code})?"
            ],
            "course_info": [
                "Can you tell me about {course_name} ({course_code})?",
                "What is {course_name} about?",
                "What does {course_code} cover?",
                "Give me information about {course_name} ({course_code})"
            ],
            "course_details": [
                "How many credits is {course_name} ({course_code})?",
                "What is the ECTS value of {course_code}?",
                "Which department offers {course_name}?",
                "What faculty is {course_code} part of?"
            ]
        }
        
    def generate_paraphrase(self, text: str) -> str:
        """
        Generate a natural answer based on the formatted information.
        
        Args:
            text (str): Formatted course information
            
        Returns:
            str: Natural language answer
        """
        # Create a prompt that encourages natural language generation
        input_text = f"generate answer: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            do_sample=True,
            temperature=0.7,
            no_repeat_ngram_size=2,
            top_p=0.9,
            top_k=50
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If the model fails to generate a good answer, return the formatted text
        if not answer or answer.lower() in ["false", "true", "none"]:
            return text
            
        return answer
    
    def generate_consistent_pair(self, course_info: Dict, query_type: str = "prerequisites") -> Tuple[str, str]:
        """
        Generate a consistent pair of questions for the same course information.
        
        Args:
            course_info (Dict): Dictionary containing course information
            query_type (str): Type of query (prerequisites, course_info, course_details)
            
        Returns:
            Tuple[str, str]: A pair of consistent questions
        """
        # Extract key information
        course_name = course_info.get("name", "")
        course_code = course_info.get("code", "")
        
        # Select templates based on query type
        templates = self.question_templates.get(query_type, self.question_templates["prerequisites"])
        
        # Generate two different questions using the templates
        questions = random.sample(templates, 2)
        
        # Format questions with course information
        q1 = questions[0].format(course_name=course_name, course_code=course_code)
        q2 = questions[1].format(course_name=course_name, course_code=course_code)
        
        return q1, q2
    
    def format_course_info(self, course: Dict, query_type: str = "prerequisites") -> str:
        """
        Format course information into a comprehensive answer based on query type.
        
        Args:
            course (Dict): Course information dictionary
            query_type (str): Type of query (prerequisites, course_info, course_details)
            
        Returns:
            str: Formatted course information
        """
        if query_type == "prerequisites":
            return self._format_prerequisites(course)
        elif query_type == "course_info":
            return self._format_course_info(course)
        elif query_type == "course_details":
            return self._format_course_details(course)
        else:
            return self._format_prerequisites(course)  # Default to prerequisites
    
    def _format_prerequisites(self, course: Dict) -> str:
        """Format prerequisites information."""
        course_name = course.get("name", "")
        course_code = course.get("code", "")
        prerequisites = course.get("prerequisites", "")
        
        if prerequisites:
            return f"{course_name} ({course_code}) requires the following prerequisites: {prerequisites}"
        else:
            return f"{course_name} ({course_code}) has no prerequisites."
    
    def _format_course_info(self, course: Dict) -> str:
        """Format comprehensive course information."""
        course_name = course.get("name", "")
        course_code = course.get("code", "")
        description = course.get("description", "")
        credits = course.get("credits", "")
        ects = course.get("ects", "")
        department = course.get("department", "")
        faculty = course.get("faculty", "")
        
        info_parts = [
            f"{course_name} ({course_code}) is a {credits}-credit course worth {ects} ECTS credits.",
            f"It is offered by the {department} in the {faculty}.",
            f"The course covers {description}"
        ]
        
        return " ".join(info_parts)
    
    def _format_course_details(self, course: Dict) -> str:
        """Format specific course details."""
        course_name = course.get("name", "")
        course_code = course.get("code", "")
        credits = course.get("credits", "")
        ects = course.get("ects", "")
        department = course.get("department", "")
        faculty = course.get("faculty", "")
        
        return f"{course_name} ({course_code}) is a {credits}-credit course worth {ects} ECTS credits. It is offered by the {department} in the {faculty}."
    
    def augment_dataset(self, course_data: List[Dict], num_pairs: int = 1000) -> List[Tuple[str, str, str]]:
        """
        Generate augmented training pairs using CGDA.
        
        Args:
            course_data (List[Dict]): List of course information dictionaries
            num_pairs (int): Number of pairs to generate
            
        Returns:
            List[Tuple[str, str, str]]: List of (question1, question2, answer) tuples
        """
        augmented_pairs = []
        query_types = ["prerequisites", "course_info", "course_details"]
        
        for _ in tqdm(range(num_pairs), desc="Generating CGDA pairs"):
            # Randomly select a course and query type
            course = random.choice(course_data)
            query_type = random.choice(query_types)
            
            # Generate consistent question pair
            q1, q2 = self.generate_consistent_pair(course, query_type)
            
            # Generate comprehensive answer
            formatted_info = self.format_course_info(course, query_type)
            answer = self.generate_paraphrase(formatted_info)
            
            augmented_pairs.append((q1, q2, answer))
        
        return augmented_pairs
    
    def save_augmented_data(self, augmented_pairs: List[Tuple[str, str, str]], output_file: str):
        """
        Save augmented data to a file.
        
        Args:
            augmented_pairs (List[Tuple[str, str, str]]): List of augmented pairs
            output_file (str): Path to output file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for q1, q2, answer in augmented_pairs:
                f.write(f"Q1: {q1}\n")
                f.write(f"Q2: {q2}\n")
                f.write(f"A: {answer}\n")
                f.write("-" * 80 + "\n")

if __name__ == "__main__":
    # Example usage
    augmentor = CGDAugmentor()
    
    # Sample course data
    sample_course = {
        "code": "CET101",
        "name": "Introduction to Educational Technology",
        "turkish_name": "Eğitim Teknolojisine Giriş",
        "credits": "4",
        "ects": "9",
        "description": "Introduction to the discipline of educational technology; discussion of constitutive problems of education and technology; developing a critical perspective on the relationship of education and technology; analysis of technologies with educational implications.",
        "prerequisites": "",
        "department": "Bilgisayar ve Öğretim Teknolojileri Eğitimi Bölümü",
        "faculty": "Eğitim Fakültesi"
    }
    
    # Test different query types
    for query_type in ["prerequisites", "course_info", "course_details"]:
        print(f"\nTesting {query_type} queries:")
        q1, q2 = augmentor.generate_consistent_pair(sample_course, query_type)
        print(f"Question 1: {q1}")
        print(f"Question 2: {q2}")
        
        formatted_info = augmentor.format_course_info(sample_course, query_type)
        paraphrase = augmentor.generate_paraphrase(formatted_info)
        print(f"Original: {formatted_info}")
        print(f"Paraphrase: {paraphrase}") 