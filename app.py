import gradio as gr
from transformers import pipeline

# Load sentiment analysis model
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    if not text:
        return "Please enter some text"
    
    result = classifier(text)[0]
    sentiment = f"{result['label']}: {result['score']:.2%} confidence"
    return sentiment

# Create interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here..."),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Analysis",
    description="Analyze the sentiment of any text",
    examples=[
        ["I love this product!"],
        ["This is terrible."],
        ["It's okay, nothing special."]
    ]
)

demo.launch()
