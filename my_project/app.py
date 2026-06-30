import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

class_names = ["pizza", "steak", "sushi"]

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=len(class_names), seed=42)

effnetb2.load_state_dict(torch.load("09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth"))

def predict(img) -> Tuple[Dict,float] :

    start_time = timer()

    img = effnetb2_transforms(img).unsqueeze(0)

    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time

example_list = [["examples/" + filepath] for filepath in os.listdir("examples")]

title = "My first NN project."
description = "I followed mrdbourke's tutorial and created this programme that classifies images of pizza, steak, sushi!"
article = "If you want to know how to create it, please visit this website. [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time(s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

demo.launch()