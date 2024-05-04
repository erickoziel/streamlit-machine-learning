import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import ImageDraw
from PIL import Image 
from google.cloud import vision
import io 
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'private/project-actas-ine.json' # Credentials

# st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="EOHD Test", page_icon="📜", layout="centered", initial_sidebar_state="expanded")

# Custom prediction function
def predict_coordinates(model, image):
    prediction = model.predict(np.array([image]))
    return np.round(prediction[0], 0).astype(int)

# https://cloud.google.com/vision/docs/handwriting
# https://www.youtube.com/watch?v=kZ3OL3AN_IA&t=205s
# https://www.youtube.com/watch?v=ddWRX2Y71RU
def detect_document(image_binary):
    """Detects document features in an image."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    # with open(path, "rb") as image_file:
    #     content = image_file.read()

    image = vision.Image(content=image_binary)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print(f"\nBlock confidence: {block.confidence}\n")

            for paragraph in block.paragraphs:
                print("Paragraph confidence: {}".format(paragraph.confidence))

                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    print(
                        "Word text: {} (confidence: {})".format(
                            word_text, word.confidence
                        )
                    )

                    for symbol in word.symbols:
                        print(
                            "\tSymbol: {} (confidence: {})".format(
                                symbol.text, symbol.confidence
                            )
                        )

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    # Return the extracted text with paragraph breaks
    print("EXTRACTED TEXT: ", response.full_text_annotation.text)
    return response.full_text_annotation.text

css = '''
<style>



    /*
    // ===============================
    // Hide top space 
    // ===============================
    */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /*
    // ===============================
    // Hide close button in sidebar
    // ===============================
    */
    button[data-testid="baseButton-header"] svg {
        display: none;
    }

    /*
    // ===============================
    // Hide titles link symbol
    // https://discuss.streamlit.io/t/hide-titles-link/19783/12
    // ===============================      
    */
    /* Hide the link button only in main area and not in all .stApp */
    .main a:first-child {
        display: none;
    }

    /*
            // ===============================
            // Hide Streamlit elements 
            // ===============================
            */
            div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
            }
            div[data-testid="stDecoration"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
            }
            div[data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
            }
            #MainMenu {
            visibility: hidden;
            height: 0%;
            }
            header {
            visibility: hidden;
            height: 0%;
            }
            footer {
            visibility: hidden;
            height: 0%;
            }

    /* Change default file uploader style */
    [data-testid='stFileUploader'] {
        width: max-content;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
        display: none;
    }

</style>
'''
st.markdown(css, unsafe_allow_html=True)



# Load model only once
@st.cache_resource
def get_model():
    model = load_model('models/model.h5')
    return model

model = get_model()

# Encabezado de la app
st.write("""
# Procesamiento de acta
Esta es una aplicación para procesar actas de manera automática.""")
# Sidebar
st.sidebar.header('Datos de entrada')

uploaded_image = st.sidebar.file_uploader('Sube una imagen', type=['jpg', 'png']) 

if not uploaded_image:
    st.info('Por favor sube una imagen en el panel de la izquierda.')
    st.stop()

processed_finished = False

if uploaded_image is not None: 
    img = Image.open(uploaded_image) 
    st.sidebar.success(f'{uploaded_image.name}')

    # image_element = st.image(img)
    # print image name

    def process_image(target_size, img):
        # Preprocess the image
        target_size = (700, 700)
        
        original_width, original_height = img.size
        scale_x = target_size[0] / original_width if target_size else 1
        scale_y = target_size[1] / original_height if target_size else 1
        
        # Resize the image
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convertir la imagen a un array numpy y normalizar los valores
        img_array = np.array(img_resized) / 255.0

        return img_array 

    def predict_coordinates(model, img_array):
        # Convert the image to an array
        prediction = model.predict(np.array([img_array]))
        pred_coordinates = np.round(prediction[0], 0).astype(int)
        return pred_coordinates

    def descale_coordinates(pred_coordinates, img, target_size):

        original_width, original_height = img.size

        scaled_coords = pred_coordinates
        original_image_size = (original_width, original_height)
        scaled_image_size = target_size

        scale_x = original_image_size[0] / scaled_image_size[0]
        scale_y = original_image_size[1] / scaled_image_size[1]

        original_xmin = int(round(scaled_coords[0] * scale_x, 0))
        original_ymin = int(round(scaled_coords[1] * scale_y, 0))
        original_xmax = int(round(scaled_coords[2] * scale_x, 0))
        original_ymax = int(round(scaled_coords[3] * scale_y, 0))

        descaled_coords = [original_xmin, original_ymin, original_xmax, original_ymax]

        return descaled_coords
    
    # Preprocess the image
    target_size = (700, 700)
    img_array = process_image(target_size, img)

    # Predict the coordinates
    pred_coordinates = predict_coordinates(model, img_array)

    # Descale the coordinates
    descaled_coords = descale_coordinates(pred_coordinates, img, target_size)

    # Create a rectangle image
    img_rectangle = img.copy()
    draw = ImageDraw.Draw(img_rectangle)
    draw.rectangle(descaled_coords, outline='red', width=10)

    st.subheader('Imagen con coordenadas estimadas')
    image_element = st.empty()
    image_element = st.image(img_rectangle)

    st.info(f"Las coordenadas estimadas son: {descaled_coords}")

    # Show the extracted image from descaled coordinates
    st.subheader('Imgen recortada con las coordenadas estimadas')
    cropped_image = img.crop(descaled_coords)
    st.image(cropped_image)

    # Google api to extract text from the image
    st.subheader('Texto extraído')

    img_byte_arr = io.BytesIO()
    cropped_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    with st.spinner('Extrayendo texto...'):
        extracted_text = detect_document(img_byte_arr)
        st.info(extracted_text.replace("\n", "  \n"))

        st.write(" ")
        st.write(" \n ")

    

