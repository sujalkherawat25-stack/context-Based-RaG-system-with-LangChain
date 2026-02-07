

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Poetry (Follow this [Poetry installation tutorial](https://python-poetry.org/docs/#installation) to install Poetry on your system)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/raj11kanani/LLM_RAG_Document_QNA
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install --no-root
   ```

3. Set up your environment variables:

   -  `.env` and insert API_KEYS variables inside with your own values. 


4. Activate the Poetry shell to run the examples:

   ```bash
   poetry shell
   ```

## Repository Structure

Here's a breakdown of the folders and what you'll find in each:

### 1. Chat Models

- `embedding.py`
   - creating embedding of your .txt file and store as vector db
- `app.py`
   - main file to execute continue_chat app 

