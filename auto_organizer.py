import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from collections import Counter

# Initialize session state for clustering
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3

# Page header
st.title("🗂️ Auto Document Organizer")
st.markdown("### AI-Powered Document Categorization")

# Check prerequisites
if 'tfidf_matrix' not in st.session_state or st.session_state.tfidf_matrix is None:
    st.warning("⚠️ No documents analyzed yet!")
    
    if st.button("📤 Go to Upload & Analyze"):
        st.switch_page("upload_analyze.py")
    st.stop()

# Explanation
with st.expander("ℹ️ How Auto-Organization Works"):
    st.markdown("""
    ### K-Means Clustering
    
    Our system uses **K-Means clustering** algorithm to automatically group similar documents:
    
    1. **Analyze Content**: Convert documents to numerical vectors (TF-IDF)
    2. **Find Patterns**: Identify natural groupings in document content
    3. **Create Categories**: Assign each document to most relevant cluster
    4. **Name Clusters**: Automatically suggest category names
    
    ### Use Cases
    
    - **Research Papers**: Group by methodology, topic, or field
    - **News Articles**: Organize by subject (politics, sports, tech)
    - **Product Reviews**: Category by product type
    - **Legal Documents**: Sort by contract type or case category
    - **Blog Posts**: Categorize by content theme
    
    ### Benefits
    
    - ⚡ **Instant**: Process 100+ documents in seconds
    - 🎯 **Smart**: Learns patterns from content
    - 📊 **Flexible**: Choose number of categories
    - 🔄 **Adaptive**: Works with any document type
    """)

st.divider()

# Configuration
st.subheader("⚙️ Organization Settings")

col1, col2 = st.columns(2)

with col1:
    # Number of categories
    n_clusters = st.slider(
        "Number of Categories",
        min_value=2,
        max_value=min(10, len(st.session_state.documents)),
        value=min(3, len(st.session_state.documents)),
        help="How many categories should documents be organized into?"
    )
    
    st.session_state.n_clusters = n_clusters

with col2:
    # Organization mode
    org_mode = st.radio(
        "Organization Mode",
        ["Automatic Naming", "Custom Category Names"],
        help="Let AI suggest names or define your own"
    )

st.divider()

# Custom category names (if selected)
category_names = {}
if org_mode == "Custom Category Names":
    st.markdown("#### 📝 Define Category Names")
    
    cols = st.columns(min(3, n_clusters))
    for i in range(n_clusters):
        with cols[i % 3]:
            category_names[i] = st.text_input(
                f"Category {i + 1}",
                value=f"Category {i + 1}",
                key=f"cat_name_{i}"
            )

# Organize button
if st.button("🚀 Organize Documents", type="primary", use_container_width=True):
    with st.spinner("Organizing documents..."):
        # Verify data consistency
        if st.session_state.tfidf_matrix.shape[0] != len(st.session_state.documents):
            st.error("⚠️ Data mismatch detected. Please re-analyze your documents.")
            if st.button("🔄 Go to Upload & Analyze"):
                st.switch_page("upload_analyze.py")
            st.stop()
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(st.session_state.tfidf_matrix.toarray())
        
        st.session_state.clusters = clusters
        
        # Generate automatic labels if needed
        if org_mode == "Automatic Naming":
            cluster_labels = {}
            
            for cluster_id in range(n_clusters):
                # Get documents in this cluster
                cluster_docs = [i for i, c in enumerate(clusters) if c == cluster_id]
                
                # Get top terms for this cluster
                cluster_vectors = st.session_state.tfidf_matrix[cluster_docs].toarray()
                avg_vector = np.mean(cluster_vectors, axis=0)
                
                # Get feature names
                feature_names = st.session_state.vectorizer.get_feature_names_out()
                
                # Top terms
                top_indices = np.argsort(avg_vector)[-3:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                
                # Create label
                cluster_labels[cluster_id] = f"{', '.join(top_terms).title()}"
        else:
            cluster_labels = category_names
        
        st.session_state.cluster_labels = cluster_labels
        
        st.success("✅ Documents organized successfully!")
        st.rerun()

# Display results
if st.session_state.clusters is not None:
    st.divider()
    st.subheader("📊 Organization Results")
    
    # Summary metrics
    cluster_counts = Counter(st.session_state.clusters)
    
    metric_cols = st.columns(min(4, n_clusters))
    for i in range(n_clusters):
        with metric_cols[i % 4]:
            st.metric(
                st.session_state.cluster_labels[i],
                f"{cluster_counts[i]} docs",
                help=f"Category {i + 1}"
            )
    
    st.divider()
    
    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📋 Categorized List", "📊 Category Analysis", "🗺️ Visual Map"])
    
    with viz_tab1:
        st.markdown("### Documents by Category")
        
        # Display documents by cluster
        for cluster_id in range(n_clusters):
            cluster_docs = [(i, st.session_state.documents[i]) for i, c in enumerate(st.session_state.clusters) if c == cluster_id]
            
            if cluster_docs:
                with st.expander(f"📁 {st.session_state.cluster_labels[cluster_id]} ({len(cluster_docs)} documents)", expanded=True):
                    for idx, doc in cluster_docs:
                        with st.container(border=True):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**📄 {doc['name']}**")
                                preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                                st.caption(preview)
                            
                            with col2:
                                st.write(f"📊 {len(doc['content'].split())} words")
                                
                                # Show similarity to cluster center
                                cluster_sim = st.session_state.similarity_matrix[idx, [i for i, c in enumerate(st.session_state.clusters) if c == cluster_id and i != idx]]
                                if len(cluster_sim) > 0:
                                    avg_sim = np.mean(cluster_sim)
                                    st.write(f"🎯 {avg_sim * 100:.0f}% match")
    
    with viz_tab2:
        st.markdown("### Category Statistics")
        
        # Create statistics dataframe
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_docs = [i for i, c in enumerate(st.session_state.clusters) if c == cluster_id]
            
            # Average document length
            avg_length = np.mean([len(st.session_state.documents[i]['content'].split()) for i in cluster_docs])
            
            # Intra-cluster similarity (cohesion)
            if len(cluster_docs) > 1:
                similarities = []
                for i in cluster_docs:
                    for j in cluster_docs:
                        if i < j:
                            similarities.append(st.session_state.similarity_matrix[i, j])
                cohesion = np.mean(similarities) if similarities else 0
            else:
                cohesion = 1.0
            
            cluster_stats.append({
                'Category': st.session_state.cluster_labels[cluster_id],
                'Documents': len(cluster_docs),
                'Avg Length (words)': int(avg_length),
                'Cohesion (%)': round(cohesion * 100, 1)
            })
        
        df_stats = pd.DataFrame(cluster_stats)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        st.caption("**Cohesion**: How similar documents within each category are (higher is better)")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Category size pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=[st.session_state.cluster_labels[i] for i in range(n_clusters)],
                    values=[cluster_counts[i] for i in range(n_clusters)],
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="Documents per Category",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cohesion bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=[s['Category'] for s in cluster_stats],
                    y=[s['Cohesion (%)'] for s in cluster_stats],
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="Category Cohesion",
                xaxis_title="Category",
                yaxis_title="Cohesion (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab3:
        st.markdown("### Document-Category Map")
        
        # Create 2D visualization using similarity matrix (simplified PCA)
        # For simplicity, use first two dimensions of similarity relationships
        from sklearn.decomposition import PCA
        from sklearn.manifold import MDS
        
        # Use MDS to reduce to 2D
        try:
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            dissimilarity = 1 - st.session_state.similarity_matrix
            coords_2d = mds.fit_transform(dissimilarity)
            
            # Create scatter plot
            colors = px.colors.qualitative.Set3[:n_clusters]
            
            fig = go.Figure()
            
            for cluster_id in range(n_clusters):
                cluster_docs = [i for i, c in enumerate(st.session_state.clusters) if c == cluster_id]
                
                fig.add_trace(go.Scatter(
                    x=coords_2d[cluster_docs, 0],
                    y=coords_2d[cluster_docs, 1],
                    mode='markers+text',
                    name=st.session_state.cluster_labels[cluster_id],
                    text=[st.session_state.documents[i]['name'][:15] + '...' for i in cluster_docs],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=colors[cluster_id % len(colors)]
                    )
                ))
            
            fig.update_layout(
                title="Document Distribution by Category",
                height=600,
                showlegend=True,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Documents that are closer together are more similar. Colors represent different categories.")
            
        except Exception as e:
            st.warning("Unable to generate 2D visualization. This may happen with very similar documents.")
    
    # Export organized structure
    st.divider()
    st.subheader("💾 Export Organization")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # Export categorization
        df_export = pd.DataFrame([
            {
                'Document': st.session_state.documents[i]['name'],
                'Category': st.session_state.cluster_labels[st.session_state.clusters[i]],
                'Category_ID': st.session_state.clusters[i] + 1
            }
            for i in range(len(st.session_state.documents))
        ])
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            "📥 Download Categorization (CSV)",
            csv,
            "document_categories.csv",
            "text/csv",
            use_container_width=True
        )
    
    with export_col2:
        # Export detailed report
        detailed_report = []
        for cluster_id in range(n_clusters):
            cluster_docs = [i for i, c in enumerate(st.session_state.clusters) if c == cluster_id]
            
            for doc_idx in cluster_docs:
                detailed_report.append({
                    'Document': st.session_state.documents[doc_idx]['name'],
                    'Category': st.session_state.cluster_labels[cluster_id],
                    'Word_Count': len(st.session_state.documents[doc_idx]['content'].split()),
                    'Preview': st.session_state.documents[doc_idx]['content'][:100] + "..."
                })
        
        df_detailed = pd.DataFrame(detailed_report)
        csv_detailed = df_detailed.to_csv(index=False)
        
        st.download_button(
            "📥 Download Detailed Report (CSV)",
            csv_detailed,
            "detailed_organization_report.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Recommendations
    st.divider()
    st.subheader("💡 Recommendations")
    
    # Check for imbalanced clusters
    max_size = max(cluster_counts.values())
    min_size = min(cluster_counts.values())
    
    if max_size / min_size > 3:
        st.warning(f"""
        **⚠️ Imbalanced Categories Detected**
        
        Some categories have significantly more documents than others.
        - Largest: {max_size} documents
        - Smallest: {min_size} documents
        
        Consider:
        - Adjusting number of categories
        - Checking if documents are truly similar
        - Splitting large categories further
        """)
    
    # Check for low cohesion
    low_cohesion_clusters = [s for s in cluster_stats if s['Cohesion (%)'] < 50]
    
    if low_cohesion_clusters:
        st.info(f"""
        **ℹ️ Low Cohesion in Some Categories**
        
        {len(low_cohesion_clusters)} category/categories have low internal similarity (<50%).
        
        This might indicate:
        - Too few categories (documents are being forced together)
        - Documents don't naturally group into distinct categories
        - Mixed content in your document collection
        
        Try increasing the number of categories.
        """)
    
    # Success message
    if max_size / min_size <= 2 and not low_cohesion_clusters:
        st.success("""
        ✅ **Excellent Organization!**
        
        Your documents are well-distributed across categories with good cohesion.
        This organization structure should work well for your needs.
        """)

else:
    st.info("👆 Configure settings and click 'Organize Documents' to begin")
    
    # Preview what will happen
    st.markdown("### 🎯 What will happen:")
    st.markdown(f"""
    - Your **{len(st.session_state.documents)} documents** will be analyzed
    - Documents will be grouped into **{n_clusters} categories**
    - Each category will be automatically named based on content
    - You'll see a complete breakdown of the organization
    """)

# Navigation
st.divider()
st.markdown("### 🎯 Related Features")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("🔍 Find Similar Documents", use_container_width=True):
        st.switch_page("similarity_finder.py")

with nav_col2:
    if st.button("⚠️ Check Plagiarism", use_container_width=True):
        st.switch_page("plagiarism_checker.py")

with nav_col3:
    if st.button("📊 Bulk Comparison", use_container_width=True):
        st.switch_page("bulk_comparison.py")