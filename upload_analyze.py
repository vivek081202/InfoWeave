import streamlit as st
import time
from collections import Counter
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'similarity_matrix' not in st.session_state:
    st.session_state.similarity_matrix = None

# English stop words
ENGLISH_STOPWORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'into', 'year', 'your', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after',
    'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new',
    'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was',
    'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having', 'may', 'should'
}

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in ENGLISH_STOPWORDS and len(word) > 2]
    
    return ' '.join(tokens)

def analyze_documents():
    """Process and analyze all documents"""
    if not st.session_state.documents:
        return False
    
    # Preprocess all documents
    processed_docs = []
    for doc in st.session_state.documents:
        processed_text = preprocess_text(doc['content'])
        processed_docs.append({
            'name': doc['name'],
            'original': doc['content'],
            'processed': processed_text,
            'word_count': len(doc['content'].split()),
            'unique_words': len(set(processed_text.split()))
        })
    
    st.session_state.processed_documents = processed_docs
    
    # Create TF-IDF vectors
    texts = [doc['processed'] for doc in processed_docs]
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    st.session_state.tfidf_matrix = tfidf_matrix
    st.session_state.vectorizer = vectorizer
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    st.session_state.similarity_matrix = similarity_matrix
    
    return True

# Page header
st.title("📤 Upload & Analyze Documents")
st.markdown("### Step 1: Add Your Documents")

# Quick Stats
if st.session_state.documents:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents Loaded", len(st.session_state.documents))
    
    if st.session_state.processed_documents:
        total_words = sum(doc['word_count'] for doc in st.session_state.processed_documents)
        col2.metric("Total Words", f"{total_words:,}")
        col3.metric("Avg Words/Doc", f"{total_words // len(st.session_state.processed_documents):,}")
        col4.metric("Status", "✅ Analyzed", help="Documents are ready for similarity detection")
    else:
        col2.metric("Total Words", "-")
        col3.metric("Avg Words/Doc", "-")
        col4.metric("Status", "⏳ Pending", help="Click 'Analyze All Documents' to process")
        st.warning("⚠️ Documents not analyzed yet! Click 'Analyze All Documents' button below.")
    
    st.divider()

# Upload Options
st.subheader("📥 Add Documents")

tab1, tab2, tab3 = st.tabs(["📁 Upload Files", "📝 Paste Text", "🎯 Load Samples"])

with tab1:
    st.markdown("**Upload text files** for analysis")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload .txt files containing your documents",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        new_docs = []
        for file in uploaded_files:
            content = file.read().decode('utf-8')
            # Check if already exists
            if not any(doc['name'] == file.name for doc in st.session_state.documents):
                new_docs.append({
                    'name': file.name,
                    'content': content,
                    'source': 'upload'
                })
        
        if new_docs:
            st.session_state.documents.extend(new_docs)
            # Clear analysis data when new documents are added
            st.session_state.processed_documents = []
            st.session_state.tfidf_matrix = None
            st.session_state.vectorizer = None
            st.session_state.similarity_matrix = None
            st.success(f"✅ Added {len(new_docs)} new document(s)! Click 'Analyze All Documents' to process.")
            st.rerun()

with tab2:
    st.markdown("**Paste text directly** for quick analysis")
    
    doc_name = st.text_input("Document Name", placeholder="e.g., My Research Paper")
    doc_content = st.text_area(
        "Document Content",
        height=300,
        placeholder="Paste your text here...",
        label_visibility="collapsed"
    )
    
    if st.button("➕ Add Document", type="primary", use_container_width=True):
        if doc_name and doc_content:
            if not any(doc['name'] == doc_name for doc in st.session_state.documents):
                st.session_state.documents.append({
                    'name': doc_name,
                    'content': doc_content,
                    'source': 'paste'
                })
                # Clear analysis data when new documents are added
                st.session_state.processed_documents = []
                st.session_state.tfidf_matrix = None
                st.session_state.vectorizer = None
                st.session_state.similarity_matrix = None
                st.success(f"✅ Added '{doc_name}'! Click 'Analyze All Documents' to process.")
                st.rerun()
            else:
                st.error("❌ A document with this name already exists!")
        else:
            st.warning("⚠️ Please provide both name and content")

with tab3:
    st.markdown("**Load sample documents** to try the system")
    
    sample_category = st.selectbox(
        "Choose sample set",
        ["Technology Articles", "Research Papers", "News Articles", "Product Reviews"]
    )
    
    if st.button("📚 Load Sample Documents", type="primary", use_container_width=True):
        samples = []
        
        if sample_category == "Technology Articles":
            samples = [
                {
                    'name': 'AI and Machine Learning.txt',
                    'content': 'Artificial intelligence and machine learning are transforming industries. Deep learning algorithms can now process vast amounts of data to identify patterns and make predictions. Neural networks are becoming increasingly sophisticated, enabling applications like computer vision, natural language processing, and autonomous systems. Companies are investing heavily in AI research to gain competitive advantages. The technology promises to revolutionize healthcare, finance, transportation, and many other sectors.'
                },
                {
                    'name': 'Cloud Computing Trends.txt',
                    'content': 'Cloud computing has become essential for modern businesses. Companies are migrating their infrastructure to cloud platforms like AWS, Azure, and Google Cloud. This shift enables scalability, reduces costs, and improves accessibility. Hybrid cloud solutions are gaining popularity, combining on-premises and cloud resources. Security remains a top concern as organizations store sensitive data in the cloud. Multi-cloud strategies are emerging to avoid vendor lock-in and increase resilience.'
                },
                {
                    'name': 'Cybersecurity Best Practices.txt',
                    'content': 'Cybersecurity threats are evolving rapidly in the digital age. Organizations must implement robust security measures to protect sensitive information. Multi-factor authentication, encryption, and regular security audits are essential. Employee training is crucial as human error remains a major vulnerability. Zero-trust architecture is becoming the new standard for network security. Companies need to stay updated on the latest threats and defense mechanisms to safeguard their digital assets.'
                },
                {
                    'name': 'Blockchain Technology.txt',
                    'content': 'Blockchain technology offers decentralized and secure transaction systems. Originally developed for cryptocurrencies, blockchain now has applications in supply chain, healthcare, and finance. Smart contracts automate agreements without intermediaries. The immutable ledger ensures transparency and trust. However, scalability and energy consumption remain challenges. Many industries are exploring how blockchain can improve efficiency and reduce fraud.'
                },
                {
                    'name': 'Machine Learning Applications.txt',
                    'content': 'Machine learning algorithms are revolutionizing data analysis across industries. Supervised learning models can classify data and make predictions with high accuracy. Unsupervised learning discovers hidden patterns in datasets. Reinforcement learning enables systems to learn from experience. Applications range from recommendation systems to fraud detection. Deep learning, a subset of machine learning, uses neural networks to process complex data like images and speech.'
                }
            ]
        
        elif sample_category == "Research Papers":
            samples = [
                {
                    'name': 'Climate Change Study.txt',
                    'content': 'This research examines the impact of climate change on coastal ecosystems. Rising sea levels and increasing temperatures threaten biodiversity. Our methodology involved collecting data from multiple coastal regions over five years. Results indicate significant changes in species distribution and habitat loss. The study emphasizes the urgent need for conservation efforts and policy interventions to protect vulnerable ecosystems from climate impacts.'
                },
                {
                    'name': 'Educational Technology Research.txt',
                    'content': 'Our study investigates the effectiveness of educational technology in improving student outcomes. We conducted a randomized controlled trial with 500 students across ten schools. Digital learning tools showed significant improvements in engagement and test scores. However, the effectiveness varied based on implementation quality and teacher training. The research suggests that technology alone is not sufficient; pedagogical approaches must adapt to leverage these tools effectively.'
                },
                {
                    'name': 'Renewable Energy Analysis.txt',
                    'content': 'This paper analyzes the economic viability of renewable energy sources. Solar and wind power have become increasingly cost-competitive with fossil fuels. Our analysis includes lifecycle costs, government incentives, and environmental benefits. Results show that renewable energy can provide sustainable and economically sound alternatives. Policy recommendations include increased investment in grid infrastructure and energy storage technologies to support renewable integration.'
                }
            ]
        
        elif sample_category == "News Articles":
            samples = [
                {
                    'name': 'Tech Industry Layoffs.txt',
                    'content': 'Major technology companies announced significant workforce reductions this quarter. Over 50,000 employees across the industry face job losses. Economic uncertainty and overhiring during the pandemic are cited as primary reasons. Companies are focusing on core businesses and cutting experimental projects. The layoffs affect various departments, with some firms reducing entire divisions. Industry experts debate whether this marks a broader economic slowdown or sector-specific correction.'
                },
                {
                    'name': 'Space Exploration Milestone.txt',
                    'content': 'NASA successfully launched a new Mars rover mission yesterday. The rover carries advanced scientific instruments to search for signs of ancient life. This mission represents years of planning and international collaboration. Scientists are particularly interested in analyzing soil samples and studying Martian geology. The rover will operate for at least two years, exploring previously uncharted regions of the red planet.'
                },
                {
                    'name': 'Global Economic Outlook.txt',
                    'content': 'World economic leaders gathered to discuss global financial challenges. Inflation rates remain elevated in major economies. Central banks are carefully balancing interest rate policies to control inflation without triggering recession. Trade tensions between nations continue to create uncertainty. Economists predict moderate growth for the coming year, with regional variations. Developing nations face unique challenges in managing debt and economic recovery.'
                }
            ]
        
        else:  # Product Reviews
            samples = [
                {
                    'name': 'Smartphone Review.txt',
                    'content': 'This flagship smartphone delivers excellent performance with its latest processor. The camera system captures stunning photos in various lighting conditions. Battery life easily lasts a full day of heavy use. The display is vibrant and responsive. However, the price point is quite high compared to competitors. Overall, this phone offers premium features for users who want the best technology available.'
                },
                {
                    'name': 'Laptop Review.txt',
                    'content': 'This ultrabook combines portability with powerful performance. The sleek design makes it perfect for business travelers. Processing speed handles demanding applications smoothly. The keyboard is comfortable for extended typing sessions. Battery life exceeds manufacturer claims, lasting over 10 hours. The only drawback is limited port selection. This laptop is ideal for professionals seeking mobility without sacrificing capabilities.'
                },
                {
                    'name': 'Headphones Review.txt',
                    'content': 'These wireless headphones provide exceptional sound quality and comfort. Active noise cancellation effectively blocks ambient noise. The battery lasts approximately 30 hours on a single charge. Touch controls are intuitive and responsive. They fold compactly for travel. Sound quality rivals much more expensive models. Minor complaints include slightly plasticky build quality. Excellent value for the price point.'
                }
            ]
        
        # Add source tag
        for sample in samples:
            sample['source'] = 'sample'
        
        st.session_state.documents = samples
        st.success(f"✅ Loaded {len(samples)} sample documents!")
        st.rerun()

# Display loaded documents
if st.session_state.documents:
    st.divider()
    st.subheader(f"📚 Loaded Documents ({len(st.session_state.documents)})")
    
    # Quick actions
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Analyze All Documents", type="primary", use_container_width=True):
            with st.spinner("Analyzing documents..."):
                progress_bar = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                success = analyze_documents()
                
                if success:
                    st.success("✅ Analysis complete!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Analysis failed")
    
    with action_col2:
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            st.session_state.documents = []
            st.session_state.processed_documents = []
            st.session_state.tfidf_matrix = None
            st.session_state.vectorizer = None
            st.session_state.similarity_matrix = None
            st.success("✅ All documents cleared!")
            st.rerun()
    
    with action_col3:
        # Export document list
        if st.session_state.documents:
            df_export = pd.DataFrame([
                {'Name': doc['name'], 'Length': len(doc['content']), 'Source': doc['source']}
                for doc in st.session_state.documents
            ])
            csv = df_export.to_csv(index=False)
            st.download_button(
                "📥 Export List",
                csv,
                "document_list.csv",
                "text/csv",
                use_container_width=True
            )
    
    # Document list with preview
    st.markdown("#### 📄 Document List")
    
    for idx, doc in enumerate(st.session_state.documents):
        with st.expander(f"📄 {doc['name']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                preview = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
                st.text_area("Content Preview", preview, height=150, disabled=True, label_visibility="collapsed", key=f"preview_{idx}")
            
            with col2:
                st.markdown("**Info:**")
                st.write(f"📊 Length: {len(doc['content'])} chars")
                st.write(f"📝 Words: {len(doc['content'].split())}")
                st.write(f"📌 Source: {doc['source']}")
                
                if st.button(f"🗑️ Remove", key=f"remove_{idx}"):
                    st.session_state.documents.pop(idx)
                    if st.session_state.processed_documents:
                        st.session_state.processed_documents.pop(idx)
                    st.rerun()
    
    # Analysis Results
    if st.session_state.processed_documents:
        st.divider()
        st.subheader("📊 Analysis Results")
        
        # Statistics
        stats_tab1, stats_tab2, stats_tab3 = st.tabs(["📈 Overview", "🔤 Word Analysis", "🎯 Similarity Preview"])
        
        with stats_tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Document statistics
                df_stats = pd.DataFrame([
                    {
                        'Document': doc['name'],
                        'Original Words': doc['word_count'],
                        'Unique Terms': doc['unique_words'],
                        'Processed Length': len(doc['processed'])
                    }
                    for doc in st.session_state.processed_documents
                ])
                
                st.dataframe(df_stats, use_container_width=True, hide_index=True)
            
            with col2:
                # Summary metrics
                total_words = sum(doc['word_count'] for doc in st.session_state.processed_documents)
                total_unique = sum(doc['unique_words'] for doc in st.session_state.processed_documents)
                
                st.metric("Total Words", f"{total_words:,}")
                st.metric("Unique Terms", f"{total_unique:,}")
                st.metric("Vocabulary", len(st.session_state.vectorizer.vocabulary_))
        
        with stats_tab2:
            # Word frequency analysis
            all_words = []
            for doc in st.session_state.processed_documents:
                all_words.extend(doc['processed'].split())
            
            word_freq = Counter(all_words)
            top_words = word_freq.most_common(20)
            
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(x=[word for word, _ in top_words],
                       y=[count for _, count in top_words],
                       marker_color='lightblue')
            ])
            
            fig.update_layout(
                title="Top 20 Most Frequent Terms",
                xaxis_title="Term",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with stats_tab3:
            # Similarity matrix preview
            if st.session_state.similarity_matrix is not None:
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.similarity_matrix,
                    x=[doc['name'] for doc in st.session_state.documents],
                    y=[doc['name'] for doc in st.session_state.documents],
                    colorscale='RdYlGn',
                    text=np.round(st.session_state.similarity_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="Document Similarity Heatmap",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Green = More similar** | **Red = Less similar**. Use this to quickly spot related documents!")
        
        # Next steps
        st.divider()
        st.markdown("### ✅ Documents Ready for Analysis!")
        
        st.markdown("Choose your next action:")
        
        next_col1, next_col2, next_col3 = st.columns(3)
        
        with next_col1:
            if st.button("🔍 Find Similar Documents", use_container_width=True, type="primary"):
                st.switch_page("similarity_finder.py")
        
        with next_col2:
            if st.button("⚠️ Check Plagiarism", use_container_width=True):
                st.switch_page("plagiarism_checker.py")
        
        with next_col3:
            if st.button("🗂️ Auto-Organize", use_container_width=True):
                st.switch_page("auto_organizer.py")

else:
    st.info("👆 Upload files, paste text, or load samples to get started!")