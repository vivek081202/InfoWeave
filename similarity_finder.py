import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# Page header
st.title("🔍 Find Similar Documents")
st.markdown("### Discover Document Relationships")

# Check if documents are loaded
if 'similarity_matrix' not in st.session_state or st.session_state.similarity_matrix is None:
    st.warning("⚠️ No documents analyzed yet!")
    st.info("Please upload and analyze documents first.")
    
    if st.button("📤 Go to Upload & Analyze"):
        st.switch_page("upload_analyze.py")
    st.stop()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Find documents that are similar to each other based on content similarity.**
    
    Use cases:
    - Find duplicate or near-duplicate content
    - Discover related research papers
    - Group articles by topic
    - Identify content that needs consolidation
    """)

with col2:
    st.metric("Documents Loaded", len(st.session_state.documents))
    st.metric("Comparisons Made", len(st.session_state.documents) * (len(st.session_state.documents) - 1) // 2)

st.divider()

# Configuration
st.subheader("⚙️ Search Configuration")

col1, col2 = st.columns(2)

with col1:
    similarity_threshold = st.slider(
        "Similarity Threshold (%)",
        min_value=50,
        max_value=100,
        value=70,
        step=5,
        help="Documents with similarity above this threshold will be shown"
    )
    
    threshold_decimal = similarity_threshold / 100

with col2:
    sort_by = st.selectbox(
        "Sort Results By",
        ["Similarity (High to Low)", "Similarity (Low to High)", "Document Name"]
    )

st.divider()

# Find similar pairs
similar_pairs = []
n_docs = len(st.session_state.documents)

# Ensure similarity matrix matches document count
if st.session_state.similarity_matrix.shape[0] != n_docs:
    st.error("⚠️ Data mismatch detected. Please re-analyze your documents.")
    if st.button("🔄 Go to Upload & Analyze"):
        st.switch_page("upload_analyze.py")
    st.stop()

for i in range(n_docs):
    for j in range(i + 1, n_docs):
        similarity = st.session_state.similarity_matrix[i, j]
        if similarity >= threshold_decimal:
            similar_pairs.append({
                'Doc1_Index': i,
                'Doc1_Name': st.session_state.documents[i]['name'],
                'Doc2_Index': j,
                'Doc2_Name': st.session_state.documents[j]['name'],
                'Similarity': similarity * 100,  # Convert to percentage
                'Category': 'Very High' if similarity > 0.9 else 'High' if similarity > 0.75 else 'Moderate'
            })

# Sort results
if sort_by == "Similarity (High to Low)":
    similar_pairs.sort(key=lambda x: x['Similarity'], reverse=True)
elif sort_by == "Similarity (Low to High)":
    similar_pairs.sort(key=lambda x: x['Similarity'])
else:
    similar_pairs.sort(key=lambda x: x['Doc1_Name'])

# Display results
st.subheader(f"📊 Results: {len(similar_pairs)} Similar Pair(s) Found")

if similar_pairs:
    # Summary metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    very_high = sum(1 for p in similar_pairs if p['Category'] == 'Very High')
    high = sum(1 for p in similar_pairs if p['Category'] == 'High')
    moderate = sum(1 for p in similar_pairs if p['Category'] == 'Moderate')
    avg_similarity = np.mean([p['Similarity'] for p in similar_pairs])
    
    metric_col1.metric("Very High (>90%)", very_high)
    metric_col2.metric("High (75-90%)", high)
    metric_col3.metric("Moderate (50-75%)", moderate)
    metric_col4.metric("Average Similarity", f"{avg_similarity:.1f}%")
    
    st.divider()
    
    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📋 List View", "🗺️ Network Map", "🔥 Heatmap"])
    
    with viz_tab1:
        # List of similar pairs
        for idx, pair in enumerate(similar_pairs):
            # Color coding based on similarity
            if pair['Category'] == 'Very High':
                border_color = "🔴"
                bg_color = "#ffebee"
            elif pair['Category'] == 'High':
                border_color = "🟡"
                bg_color = "#fff9c4"
            else:
                border_color = "🟢"
                bg_color = "#e8f5e9"
            
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"### {border_color} Pair #{idx + 1}")
                    st.markdown(f"**📄 {pair['Doc1_Name']}**")
                    st.markdown(f"**📄 {pair['Doc2_Name']}**")
                
                with col2:
                    st.metric("Similarity", f"{pair['Similarity']:.1f}%")
                    st.caption(pair['Category'])
                
                # Expander for detailed comparison
                with st.expander("🔍 View Details"):
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown(f"**{pair['Doc1_Name']}**")
                        doc1_content = st.session_state.documents[pair['Doc1_Index']]['content']
                        st.text_area(
                            "Content",
                            doc1_content[:500] + "..." if len(doc1_content) > 500 else doc1_content,
                            height=200,
                            disabled=True,
                            key=f"sim_doc1_{idx}",
                            label_visibility="collapsed"
                        )
                    
                    with detail_col2:
                        st.markdown(f"**{pair['Doc2_Name']}**")
                        doc2_content = st.session_state.documents[pair['Doc2_Index']]['content']
                        st.text_area(
                            "Content",
                            doc2_content[:500] + "..." if len(doc2_content) > 500 else doc2_content,
                            height=200,
                            disabled=True,
                            key=f"sim_doc2_{idx}",
                            label_visibility="collapsed"
                        )
                    
                    # Common terms analysis
                    doc1_processed = st.session_state.processed_documents[pair['Doc1_Index']]['processed']
                    doc2_processed = st.session_state.processed_documents[pair['Doc2_Index']]['processed']
                    
                    common_words = set(doc1_processed.split()) & set(doc2_processed.split())
                    
                    if common_words:
                        st.markdown("**🔤 Common Terms:**")
                        st.caption(", ".join(sorted(list(common_words))[:30]))
    
    with viz_tab2:
        # Network visualization
        st.markdown("**Document Similarity Network**")
        st.caption("Nodes = Documents | Edges = Similarity connections | Thicker edges = More similar")
        
        # Create network graph
        G = nx.Graph()
        
        # Add all documents as nodes
        for idx, doc in enumerate(st.session_state.documents):
            G.add_node(idx, label=doc['name'])
        
        # Add edges for similar documents
        for pair in similar_pairs:
            G.add_edge(
                pair['Doc1_Index'],
                pair['Doc2_Index'],
                weight=pair['Similarity']
            )
        
        # Calculate positions
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=weight / 20,  # Scale line width
                    color=f'rgba(100, 100, 100, {weight / 100})'
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(st.session_state.documents[node]['name'])
            # Node size based on number of connections
            node_size.append(20 + G.degree(node) * 5)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            textfont=dict(size=10),
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Network statistics
        st.markdown("**Network Statistics:**")
        net_col1, net_col2, net_col3 = st.columns(3)
        net_col1.metric("Total Connections", G.number_of_edges())
        net_col2.metric("Isolated Documents", sum(1 for node in G.nodes() if G.degree(node) == 0))
        net_col3.metric("Most Connected", max(G.degree(node) for node in G.nodes()) if G.number_of_nodes() > 0 else 0)
    
    with viz_tab3:
        # Full similarity heatmap
        st.markdown("**Complete Similarity Matrix**")
        
        fig = go.Figure(data=go.Heatmap(
            z=st.session_state.similarity_matrix * 100,  # Convert to percentage
            x=[doc['name'] for doc in st.session_state.documents],
            y=[doc['name'] for doc in st.session_state.documents],
            colorscale='RdYlGn',
            text=np.round(st.session_state.similarity_matrix * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 8},
            colorbar=dict(title="Similarity %")
        ))
        
        fig.update_layout(
            title="All Document Similarities",
            height=600,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Export options
    st.subheader("💾 Export Results")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # Create export dataframe
        df_export = pd.DataFrame([
            {
                'Document 1': pair['Doc1_Name'],
                'Document 2': pair['Doc2_Name'],
                'Similarity (%)': round(pair['Similarity'], 2),
                'Category': pair['Category']
            }
            for pair in similar_pairs
        ])
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            "📥 Download Similar Pairs (CSV)",
            csv,
            "similar_documents.csv",
            "text/csv",
            use_container_width=True
        )
    
    with export_col2:
        # Export full similarity matrix
        df_matrix = pd.DataFrame(
            st.session_state.similarity_matrix * 100,
            columns=[doc['name'] for doc in st.session_state.documents],
            index=[doc['name'] for doc in st.session_state.documents]
        )
        
        csv_matrix = df_matrix.to_csv()
        st.download_button(
            "📥 Download Full Matrix (CSV)",
            csv_matrix,
            "similarity_matrix.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Recommendations
    st.divider()
    st.subheader("💡 Recommendations")
    
    if very_high > 0:
        st.warning(f"""
        **⚠️ {very_high} pair(s) with very high similarity (>90%)**
        
        These documents are nearly identical. Consider:
        - Checking for duplicate content
        - Investigating potential plagiarism
        - Merging similar content
        - Keeping only one version
        """)
    
    if high > 0:
        st.info(f"""
        **ℹ️ {high} pair(s) with high similarity (75-90%)**
        
        These documents share significant content. Consider:
        - Reviewing for overlapping information
        - Cross-referencing related topics
        - Creating links between documents
        - Consolidating similar sections
        """)

else:
    st.info(f"No documents found with similarity ≥ {similarity_threshold}%")
    st.markdown("""
    **Try:**
    - Lowering the similarity threshold
    - Adding more documents
    - Using documents from similar topics
    """)

# Next steps
st.divider()
st.markdown("### 🎯 What's Next?")

next_col1, next_col2, next_col3 = st.columns(3)

with next_col1:
    if st.button("⚠️ Check for Plagiarism", use_container_width=True, type="primary"):
        st.switch_page("plagiarism_checker.py")

with next_col2:
    if st.button("🗂️ Auto-Organize Documents", use_container_width=True):
        st.switch_page("auto_organizer.py")

with next_col3:
    if st.button("📊 Bulk Comparison", use_container_width=True):
        st.switch_page("bulk_comparison.py")