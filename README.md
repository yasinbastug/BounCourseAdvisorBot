# App Project

This repository contains the code for the App Project, focusing on the generation and answering functionality.

## Project Structure

- `build_faiss.py`: Script for building FAISS index
- `build_reranker.py`: Script for building reranker model
- `cgda/`: Core generation and data analysis modules
- `CourseAdvisor-Bot/`: Course advisor bot implementation

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd app-project
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