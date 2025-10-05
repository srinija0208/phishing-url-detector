import gradio as gr
import pandas as pd
import joblib
import os

from src.components.feature_extraction import extract_features


# load trained pipeline
pipeline = joblib.load('artifacts/best_model.pkl')


# prediction function

def predict_url(url:str):

    try:

        df = pd.DataFrame({'URL':[url]})
        df = extract_features(df)

        # predict
        pred = pipeline.predict(df)[0]
        proba = pipeline.predict_proba(df)[0][1]  ## probability of phishing


        label = "Phishing" if pred==1 else "Legitimate"
        return f"Prediction: {label}\nPhishing Probability: {proba:.2f}"

    except Exception:
        return "Invalid URL or processing error. Please try again."
    

# Custom CSS for centering and button styling
custom_css = """

/* Use Flexbox on the root app element to horizontally center the content */
.gradio-app {
    display: flex;
    justify-content: center;
    width: 100%;
}

/* Apply max width and use margin auto on the container for the content block */
.gradio-container {
    max-width: 700px ;
    margin: 0 auto ; 
}

/* Styles the Analyze URL button (using the elem_id) */
#analyze_button {
    color: white ;
    border-color: #b91c1c ;
    font-weight: bold ;
}

"""

## Gradio UI using Blocks for Vertical Layout and Custom Styles

with gr.Blocks(title="Phishing URL Detector", css=custom_css) as demo:
    gr.Markdown(
        """
        # Phishing URL Detector
        Paste any url to check if it's phishing or legitimate.
        """
    )
    
    # Components are placed in a column (vertical) layout by default
    url_input = gr.Textbox(label="Enter a URL", placeholder="e.g., https://www.google.com/")

    with gr.Row():

        predict_button = gr.Button("Analyze URL", elem_id='analyze_button', scale=1)
        clear_button = gr.Button("Clear", variant="secondary", scale=1) 

    prediction_output = gr.Textbox(label="Prediction", interactive=False)
    
    # Link the button click event to the prediction function
    predict_button.click(
        fn=predict_url,
        inputs=url_input,
        outputs=prediction_output
    )

    clear_button.click(
        fn=lambda: ["", ""],
        inputs=None,
        outputs=[url_input, prediction_output]
    )

if __name__ == '__main__':
    demo.launch()