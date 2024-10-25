# ATAI Chatbot
UZH Advanced Topics in Artificial Intelligence project.

## How to run
1. put a `password.txt` file with the password for the bot in the main directory
2. put the knowledge graph at `dataset/14_graph.nt`
3. run main.py and keep it running for as long as you want the chatbot to be online

## To install pytorch with cuda
`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia`

## To install spacy model
`python -m spacy download en_core_web_sm`