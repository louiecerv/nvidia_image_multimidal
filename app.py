import os
import base64
import requests
import streamlit as st
import json

if "stream" not in st.session_state:
    st.session_state.stream = True

api_key = os.getenv("NVIDIA_VISION_KEY")
MODEL_ID = "meta/llama-3.2-90b-vision-instruct"
invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_content(chunk):
    try:
        decoded_chunk = chunk.decode('utf-8')
        json_data = decoded_chunk.split('data: ')[1]
        parsed_data = json.loads(json_data)
        content = parsed_data['choices'][0]['delta']['content']
        return content
    except json.JSONDecodeError as e:
        #ignore the error
        return ""
    
  
def main():
    stream = st.session_state.stream

    st.title("Multimodal Image Analysis with " + MODEL_ID)

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering)
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University
    """
    with st.expander("About"):
        st.text(text)

    st.write("Upload an image and select the image analysis task.")

    # File upload for image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:

        # Encode the uploaded image to base64
        base64_image = base64.b64encode(uploaded_image.getvalue()).decode('utf-8')

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image",  use_container_width=True)

    # List of image analysis tasks
    analysis_tasks = [
        "Scene Analysis: Describe the scene depicted in the image. Identify the objects present, their spatial relationships, and any actions taking place.",
        "Object Detection and Classification: Identify and classify all objects present in the image. Provide detailed descriptions of each object, including its size, shape, color, and texture.",
        "Image Captioning: Generate a concise and accurate caption that describes the content of the image.",
        "Visual Question Answering: Answer specific questions about the image, such as 'What color is the car?' or 'How many people are in the image?'",
        "Image Similarity Search: Given a query image, find similar images from a large dataset based on visual features.",
        "Image Segmentation: Segment the image into different regions corresponding to objects or areas of interest.",
        "Optical Character Recognition (OCR): Extract text from the image, such as printed or handwritten text.",
        "Diagram Understanding: Analyze a diagram (e.g., flowchart, circuit diagram) and extract its structure and meaning.",
        "Art Analysis: Describe the artistic style, subject matter, and emotional impact of an image.",
        "Medical Image Analysis: Analyze medical images (e.g., X-rays, MRIs) to detect abnormalities or diagnose diseases."
    ]

    # Task selection dropdown
    selected_task = st.selectbox("Select an image analysis task:", analysis_tasks)
    
    

    if st.button("Generate Response"):
        st.session_state.stream = st.checkbox("Begin streaming the AI response as soon as it is available.", value=True)    
        stream = st.session_state.stream

        if uploaded_image is None or selected_task == "":
            st.error("Please upload an image and select a task.")
            return

        else:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "text/event-stream" if stream else "application/json"
            }

            # Prepare the multimodal prompt
            payload = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": f'{selected_task} <img src="data:image/png;base64,{base64_image}" />'
                    }
                ],
                "max_tokens": 512,
                "temperature": 1.00,
                "top_p": 1.00,
                "stream": stream  
            }

            with st.spinner("Processing..."):
                response = requests.post(
                    invoke_url,
                    headers=headers,
                    json=payload,
                    stream=stream  # Important for streaming
                )

                print(f"Stream: {stream}")
                
                if stream:
                    print(f"response: {response.text}")

                    response_container = st.empty()
                    content = ""
                    # Efficiently handle streaming response
                    for chunk in response.iter_lines(): 
                        
                        if len(chunk) > 0:                             
                            content += extract_content(chunk)
                            response_container.markdown(content)

                else:
                    try:
                        content = response.json()
                        content_string = content.get('choices', [{}])[0].get('message', {}).get('content', '')
                        st.write(f"AI Response: {content_string}")

                        st.success("Response generated!")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
 
if __name__ == "__main__":
    main()