## Description
A simple example in which data from the vision and speech models are fed to the LLM along with an instruction template. The template's purpose is to define the role of the LLM and instruct to use vision model results to assist the user.

The app front-end is a Gradio interface in which the user is able to record a message and attach an image to submit. That message is then transcribed and sent to the LLM\Vision wrapper. Using the image captioning model, a summary of what the image entails is generated and then fed to the LLM along with the template and previous messages.