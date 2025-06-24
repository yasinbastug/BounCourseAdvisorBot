from program_augmentation import ProgramAugmentor
from db_connection import DatabaseConnection
import os
from typing import List, Dict

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize database connection
    db = DatabaseConnection()
    
    try:
        # Load program data from MongoDB
        program_data = db.get_all_programs()
        print(f"Loaded {len(program_data)} programs from database")
        
        # Initialize Program augmentor
        augmentor = ProgramAugmentor()
        
        # Generate augmented pairs
        print("Generating Program CGDA pairs...")
        augmented_pairs = augmentor.augment_dataset(program_data, num_pairs=1000)
        
        # Save augmented data
        output_file = 'data/program_cgda_pairs.txt'
        augmentor.save_augmented_data(augmented_pairs, output_file)
        print(f"Saved {len(augmented_pairs)} Program CGDA pairs to {output_file}")
        
        # Print some examples
        print("\nExample Program CGDA pairs:")
        for i, (q1, q2, answer) in enumerate(augmented_pairs[:3]):
            print(f"\nPair {i+1}:")
            print(f"Q1: {q1}")
            print(f"Q2: {q2}")
            print(f"A: {answer}")
            
    finally:
        # Close database connection
        db.close()

if __name__ == "__main__":
    main() 