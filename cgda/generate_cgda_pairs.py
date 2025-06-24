from data_augmentation import CGDAugmentor
import os
from typing import List, Dict
from config import MONGO_URI, DB_NAME, COURSE_COLL
import pymongo

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize database connection
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COURSE_COLL]
    
    try:
        # Load course data from MongoDB
        course_data = collection.find()
        course_data = list(course_data)
        print(f"Loaded {len(course_data)} courses from database")
        
        # Initialize CGDA augmentor
        augmentor = CGDAugmentor()
        
        # Generate augmented pairs
        print("Generating CGDA pairs...")
        augmented_pairs = augmentor.augment_dataset(course_data, num_pairs=2000)
        
        # Save augmented data
        output_file = 'data/cgda_pairs.txt'
        augmentor.save_augmented_data(augmented_pairs, output_file)
        print(f"Saved {len(augmented_pairs)} CGDA pairs to {output_file}")
        
        # Print some examples
        print("\nExample CGDA pairs:")
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