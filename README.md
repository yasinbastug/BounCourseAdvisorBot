# Boğaziçi University Course Advisor Bot

This repository contains the code for the Course Advisor Bot, which helps students find and get information about courses.

## Project Structure

- `build_faiss.py`: Script for building FAISS index
- `build_reranker.py`: Script for building reranker model
- `cgda/`: Core generation and data analysis modules
- `CourseAdvisor-Bot/`: Course advisor bot implementation

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yasinbastug/BounCourseAdvisorBot.git
cd BounCourseAdvisorBot
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Linux/Mac
# or
.\env\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
MONGO_URI=your_mongodb_connection_string_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Required Data and Models

Before running the project, you'll need to:

1. Download necessary data files and place them in the `data/` directory
2. Download or train required models and place them in appropriate directories:
   - Fine-tuned models in `gen_finetuned/`
   - LoRA models in `gen_lora/`
   - Pre-trained models in `gen_pretrained/`
   - Reranker models in `reranker_tuned/`

Note: Data files and model weights are not included in this repository due to size constraints.

## Usage

[Add usage instructions here once we confirm the main entry points and commands]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 