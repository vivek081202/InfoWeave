import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="InfoWeave - Smart Document Organizer & Similarity Detector",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = {
    "App Navigation": [
        st.Page("home.py", title="Home", icon='🏠', default=True),
    ],
    "Core Features": [
        st.Page("upload_analyze.py", title="Upload & Analyze Documents", icon='📤'),
        st.Page("similarity_finder.py", title="Find Similar Documents", icon='🔍'),
        st.Page("plagiarism_checker.py", title="Plagiarism Detection", icon='⚠️'),
        st.Page("auto_organizer.py", title="Auto Document Organizer", icon='🗂️'),
    ],
    "Advanced": [
        st.Page("bulk_comparison.py", title="Bulk Document Comparison", icon='📊'),
    ]
}

pg = st.navigation(pages)
pg.run()