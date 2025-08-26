# Sentinel-2 AI Compressor

## I - Installation

1. Clone the repository:

```bash
git clone git@github.com:sebastien-tetaud/sentinel-2-ai-compressor.git
cd sentinel-2-ai-compressor
```

2. Create and activate a conda environment:

```bash
conda create -n ai_compressor python==3.13.2
conda activate ai_compressor
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your credentials by creating a `.env` file in the root directory with the following content:

```bash
touch .env
```
then:

```
ACCESS_KEY_ID=username
SECRET_ACCESS_KEY=password
```

## II - Download Dataset

```bash
cd src/generate_dataset
```

then

```bash
python download_v4.py
```

## III - Train the Model

```bash
python main.py
```

The **main.py** is using **src/cfg/config.yaml** configuration file.