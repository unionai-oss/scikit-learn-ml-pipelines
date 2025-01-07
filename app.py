'''
This is a gradio app that uses the actors environment to make predictions.
Real time & app serving is coming soon! Schedule a demo to see more. 
'''

import gradio as gr
from union.remote import UnionRemote

# Create a remote connection
remote = UnionRemote()

def predict_with_actor(pred_data):
    pred_data = [[float(i) for i in pred_data.split(",")]]
    inputs = {"pred_data": pred_data,}
    
    workflow = remote.fetch_workflow(name="workflows.workflows.actor_prediction_knn")
    execution = remote.execute(workflow, inputs=inputs, wait=True)
    print(execution.outputs['o0'])
    return execution.outputs['o0']


# Launch Gradio app
iface = gr.Interface(
    fn=predict_with_actor,
    inputs=["text"],
    outputs=gr.Textbox(label="Predictions:"),  # Change output to HTML for better formatting
    live=False,
)

iface.launch(debug=True)

# def greet(name):
#     return "Hello " + name + "!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()   

