import streamlit as st

# app de streamlit que utiliza el moodelo de ML de clasificación de flores iris 


st.title("Machine Learning Applications")
st.write("This web app showcases different machine learning applications.")

# Sidebar
st.sidebar.header('')

# show 3 columns with a giant emoji in each one
col1, col2, col3 = st.columns(3)

with col1:
    # st.header("Classification")
    _, mid_col, _ = st.columns([.3, .4, .3])
    with mid_col:
        st.title("🌷")
    st.write("Use a model to classify flowers into different species.")
    if st.button("Classification", type="secondary", use_container_width=True):
        st.switch_page("pages/00_Classification_🌷.py")
    # st.link_button("Go to page", "Classification_🌷") # it doesnt allow to open on same page because it would reload everything 
    # st.markdown('<a href="/Classification_🌷" target="_self">View all</a>',unsafe_allow_html=True)
    # st.button("Ir al proyecto", on_click="location.href='https://google.com';", type="secondary", use_container_width=True)
    
with col2:
    # st.header("Regression")
    _, mid_col, _ = st.columns([.3, .4, .3])
    with mid_col:
        st.title("🏠")
    st.write("Use a model to predict house prices.")
    if st.button("Regression", type="secondary", use_container_width=True):
        st.switch_page("pages/00_Classification_🌷.py")
    #st.link_button("Go to page", "Regression_🏠")

with col3:
    # st.header("Clustering")
    _, mid_col, _ = st.columns([.3, .4, .3])
    with mid_col:
        st.title("🛒")
    st.write("Learn more about the datasets used in this app.")
    if st.button("Clustering", type="secondary", use_container_width=True):
        st.switch_page("pages/00_Classification_🌷.py")
    #st.link_button("Go to page", "Clustering_🛒")


