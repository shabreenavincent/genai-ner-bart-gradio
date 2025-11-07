## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that involves identifying and classifying key entities like names, organizations, locations, and dates in a given text. The goal of this project is to create a user-friendly NER tool that integrates a fine-tuned BART model to demonstrate state-of-the-art capabilities in recognizing entities from textual data.
### DESIGN STEPS:

#### STEP 1: Data Collection and Preprocessing
* Collect a labeled dataset for NER tasks. Common datasets include CoNLL-2003, OntoNotes, or a custom dataset.
* Download or create a dataset with entities labeled in BIO format (Begin, Inside, Outside).
* Preprocess the text data, tokenizing it for compatibility with BART.
* Split the data into training, validation, and testing sets.
#### STEP 2: Fine-Tuning the BART Model
* Use the Hugging Face transformers library.
* Load a pre-trained BART model (facebook/bart-base or similar).
* Modify the model for token classification by adding a classification head.
* Train the model on the preprocessed dataset using a suitable optimizer and scheduler.
#### STEP 3: Model Evaluation
* Use metrics like F1-score, precision, and recall for evaluation.
* Test the model on unseen data and analyze its performance on different entity types.
#### STEP 4: Application Development Using Gradio
* Design the interface with Gradio to allow users to input text and view extracted entities.
* Integrate the fine-tuned BART model into the Gradio app.
* Define a backend function that processes user input through the model and displays the results.
#### STEP 5 :Deployment and Testing
* Host the application on a cloud platform like Hugging Face Spaces or Google Colab.
* Collect user feedback to improve usability and performance.
### PROGRAM:
```
developed by: Shabreena Vincent
reg no: 212222230141
```
```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']
# Helper function
import requests, json

def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))
API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "My name is Andrew, I'm building DeepLearningAI and I live in California"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)
def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is Andrew and I live in California", "My name is Poli and work at HuggingFace"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))
```
### OUTPUT:
<img width="933" height="500" alt="image" src="https://github.com/user-attachments/assets/ec367a88-4949-4b72-8ec1-a9f5c1e4c675" />
### RESULT:
Thus, developed an NER prototype application with user interaction and evaluation features, using a fine-tuned BART model deployed through the Gradio framework.
