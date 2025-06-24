import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Tuple
import random
import numpy as np
from tqdm import tqdm

class ProgramAugmentor:
    def __init__(self, model_name: str = "t5-base"):
        """
        Initialize the Program augmentor.
        
        Args:
            model_name (str): Name of the pre-trained model to use for augmentation
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Question templates for different types of queries
        self.question_templates = {
            "program_overview": [
                "Can you tell me about the {program_name} program?",
                "What is the {program_name} program about?",
                "Give me an overview of the {program_name} program",
                "What does the {program_name} program offer?"
            ],
            "program_structure": [
                "What is the structure of the {program_name} program?",
                "How is the {program_name} program organized?",
                "What are the requirements for the {program_name} program?",
                "What courses are offered in the {program_name} program?"
            ],
            "program_details": [
                "Is {program_name} a faculty or department?",
                "What type of program is {program_name}?",
                "What degrees are offered in {program_name}?",
                "What are the main features of {program_name}?"
            ]
        }
        
    def generate_paraphrase(self, text: str) -> str:
        """
        Generate a natural answer based on the formatted information.
        
        Args:
            text (str): Formatted program information
            
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
    
    def generate_consistent_pair(self, program_info: Dict, query_type: str = "program_overview") -> Tuple[str, str]:
        """
        Generate a consistent pair of questions for the same program information.
        
        Args:
            program_info (Dict): Dictionary containing program information
            query_type (str): Type of query (program_overview, program_structure, program_details)
            
        Returns:
            Tuple[str, str]: A pair of consistent questions
        """
        # Extract key information
        program_name = program_info.get("name", "")
        
        # Select templates based on query type
        templates = self.question_templates.get(query_type, self.question_templates["program_overview"])
        
        # Generate two different questions using the templates
        questions = random.sample(templates, 2)
        
        # Format questions with program information
        q1 = questions[0].format(program_name=program_name)
        q2 = questions[1].format(program_name=program_name)
        
        return q1, q2
    
    def format_program_info(self, program: Dict, query_type: str = "program_overview") -> str:
        """
        Format program information into a comprehensive answer based on query type.
        
        Args:
            program (Dict): Program information dictionary
            query_type (str): Type of query (program_overview, program_structure, program_details)
            
        Returns:
            str: Formatted program information
        """
        if query_type == "program_overview":
            return self._format_program_overview(program)
        elif query_type == "program_structure":
            return self._format_program_structure(program)
        elif query_type == "program_details":
            return self._format_program_details(program)
        else:
            return self._format_program_overview(program)  # Default to overview
    
    def _format_program_overview(self, program: Dict) -> str:
        """Format program overview information."""
        program_name = program.get("name", "")
        description = program.get("description", "")
        is_faculty = program.get("is_faculty", False)
        
        program_type = "Faculty" if is_faculty else "Department"
        
        return f"The {program_name} is a {program_type} that {description}"
    
    def _format_program_structure(self, program: Dict) -> str:
        """Format program structure information."""
        program_name = program.get("name", "")
        program_data = program.get("program", {})
        
        if not program_data:
            return f"The {program_name} program structure is not currently available."
        
        # Extract semester information
        semesters = []
        for semester, courses in program_data.items():
            if semester != "TOTAL":
                semester_courses = [f"{course['code']} ({course['name']})" for course in courses.values() if course['code'] != "TOTAL"]
                semesters.append(f"In {semester}, students take {', '.join(semester_courses)}")
        
        return f"The {program_name} program is structured as follows: {'; '.join(semesters)}"
    
    def _format_program_details(self, program: Dict) -> str:
        """Format specific program details."""
        program_name = program.get("name", "")
        is_faculty = program.get("is_faculty", False)
        description = program.get("description", "")
        
        program_type = "Faculty" if is_faculty else "Department"
        
        # Extract key information from description
        key_points = []
        if "bachelor's degree" in description.lower():
            key_points.append("bachelor's degree programs")
        if "master's degree" in description.lower():
            key_points.append("master's degree programs")
        if "ph.d." in description.lower():
            key_points.append("Ph.D. programs")
            
        if key_points:
            return f"The {program_name} is a {program_type} that offers {', '.join(key_points)}."
        else:
            return f"The {program_name} is a {program_type}."
    
    def augment_dataset(self, program_data: List[Dict], num_pairs: int = 1000) -> List[Tuple[str, str, str]]:
        """
        Generate augmented training pairs using CGDA.
        
        Args:
            program_data (List[Dict]): List of program information dictionaries
            num_pairs (int): Number of pairs to generate
            
        Returns:
            List[Tuple[str, str, str]]: List of (question1, question2, answer) tuples
        """
        augmented_pairs = []
        query_types = ["program_overview", "program_structure", "program_details"]
        
        for _ in tqdm(range(num_pairs), desc="Generating Program CGDA pairs"):
            # Randomly select a program and query type
            program = random.choice(program_data)
            query_type = random.choice(query_types)
            
            # Generate consistent question pair
            q1, q2 = self.generate_consistent_pair(program, query_type)
            
            # Generate comprehensive answer
            formatted_info = self.format_program_info(program, query_type)
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
    augmentor = ProgramAugmentor()
    
    # Sample program data
    sample_program = {
        "name": "Eğitim Fakültesi",
        "is_faculty": True,
        "description": "The Faculty of Education was founded in 1982, and it combines professional, academic and research objectives in line with the university's traditions of high standards. These objectives are two-fold: (a) to foster intellectual preparation and development, and (b) to provide professional training for young people seeking careers in the fields of teaching as well as guidance and psychological counseling in primary and secondary schools.",
        "program": {
            "First Semester": {
                "AE111*": {
                    "code": "AE111*",
                    "name": "Critical Skills in English I/ HSS Elective",
                    "credits": "3",
                    "ects": "3"
                }
            }
        }
    }
    
    # Test different query types
    for query_type in ["program_overview", "program_structure", "program_details"]:
        print(f"\nTesting {query_type} queries:")
        q1, q2 = augmentor.generate_consistent_pair(sample_program, query_type)
        print(f"Question 1: {q1}")
        print(f"Question 2: {q2}")
        
        formatted_info = augmentor.format_program_info(sample_program, query_type)
        paraphrase = augmentor.generate_paraphrase(formatted_info)
        print(f"Original: {formatted_info}")
        print(f"Paraphrase: {paraphrase}") 