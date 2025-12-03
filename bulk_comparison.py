import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# Page header
st.title("📊 Bulk Document Comparison")
st.markdown("### Comprehensive Analysis Dashboard")

# Check prerequisites
if 'similarity_matrix' not in st.session_state or st.session_state.similarity_matrix is None:
    st.warning("⚠️ No documents analyzed yet!")
    
    if st.button("📤 Go to Upload & Analyze"):
        st.switch_page("upload_analyze.py")
    st.stop()

# Overview
with st.expander("ℹ️ About Bulk Comparison"):
    st.markdown("""
    This dashboard provides a comprehensive overview of all document relationships
    in your collection.
    
    ### What You'll See:
    
    - **Global Statistics**: Overall metrics for your document collection
    - **Similarity Distribution**: How similar documents are to each other
    - **Document Rankings**: Most similar and most unique documents
    - **Cluster Analysis**: Natural groupings in your content
    - **Relationship Networks**: Visual map of document connections
    
    ### Best For:
    
    - Large document collections (10+ documents)
    - Understanding overall content structure
    - Identifying outliers and duplicates
    - Quality assurance checks
    - Content audit reports
    """)

st.divider()

# Global Statistics
st.subheader("📈 Global Statistics")

n_docs = len(st.session_state.documents)

# Verify data consistency
if st.session_state.similarity_matrix.shape[0] != n_docs:
    st.error("⚠️ Data mismatch detected. Please re-analyze your documents.")
    if st.button("🔄 Go to Upload & Analyze"):
        st.switch_page("upload_analyze.py")
    st.stop()

total_comparisons = (n_docs * (n_docs - 1)) // 2

# Calculate various metrics
similarity_values = []
for i in range(n_docs):
    for j in range(i + 1, n_docs):
        similarity_values.append(st.session_state.similarity_matrix[i, j])

avg_similarity = np.mean(similarity_values) if similarity_values else 0
max_similarity = np.max(similarity_values) if similarity_values else 0
min_similarity = np.min(similarity_values) if similarity_values else 0
std_similarity = np.std(similarity_values) if similarity_values else 0

# Display metrics
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Documents", n_docs)
col2.metric("Comparisons", f"{total_comparisons:,}")
col3.metric("Avg Similarity", f"{avg_similarity * 100:.1f}%")
col4.metric("Max Similarity", f"{max_similarity * 100:.1f}%")
col5.metric("Std Deviation", f"{std_similarity * 100:.1f}%")

# Interpretation
if avg_similarity > 0.7:
    st.warning("⚠️ **High average similarity** - Your documents are very similar. Consider more diverse content or check for duplicates.")
elif avg_similarity < 0.3:
    st.info("ℹ️ **Low average similarity** - Your documents are quite diverse. Good for varied content libraries.")
else:
    st.success("✅ **Moderate similarity** - Balanced mix of similar and diverse content.")

st.divider()

# Detailed Analysis Tabs
analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
    "📊 Similarity Distribution",
    "🏆 Document Rankings",
    "🗺️ Network Analysis",
    "📋 Detailed Matrix"
])

with analysis_tab1:
    st.markdown("### Similarity Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Histogram of similarities
        fig = go.Figure(data=[
            go.Histogram(
                x=[s * 100 for s in similarity_values],
                nbinsx=20,
                marker_color='lightblue',
                name='Similarity'
            )
        ])
        
        fig.update_layout(
            title="Distribution of Document Similarities",
            xaxis_title="Similarity (%)",
            yaxis_title="Number of Document Pairs",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Similarity Ranges:**")
        
        very_high = sum(1 for s in similarity_values if s > 0.9)
        high = sum(1 for s in similarity_values if 0.75 < s <= 0.9)
        moderate = sum(1 for s in similarity_values if 0.5 < s <= 0.75)
        low = sum(1 for s in similarity_values if s <= 0.5)
        
        st.metric("Very High (>90%)", very_high)
        st.metric("High (75-90%)", high)
        st.metric("Moderate (50-75%)", moderate)
        st.metric("Low (<50%)", low)
    
    # Box plot
    st.markdown("### Statistical Summary")
    
    fig = go.Figure(data=[
        go.Box(
            y=[s * 100 for s in similarity_values],
            name="Similarities",
            marker_color='lightgreen',
            boxmean='sd'
        )
    ])
    
    fig.update_layout(
        title="Similarity Statistics (Box Plot)",
        yaxis_title="Similarity (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with analysis_tab2:
    st.markdown("### Document Rankings")
    
    # Calculate average similarity for each document
    doc_avg_similarities = []
    for i in range(n_docs):
        similarities = []
        for j in range(n_docs):
            if i != j:
                similarities.append(st.session_state.similarity_matrix[i, j])
        
        doc_avg_similarities.append({
            'Index': i,
            'Document': st.session_state.documents[i]['name'],
            'Avg Similarity': np.mean(similarities),
            'Max Similarity': np.max(similarities),
            'Min Similarity': np.min(similarities)
        })
    
    # Rankings
    rank_tab1, rank_tab2 = st.tabs(["Most Similar", "Most Unique"])
    
    with rank_tab1:
        st.markdown("**Documents with highest average similarity to others**")
        st.caption("These documents are most representative of your collection")
        
        sorted_similar = sorted(doc_avg_similarities, key=lambda x: x['Avg Similarity'], reverse=True)
        
        for idx, doc_info in enumerate(sorted_similar[:10], 1):
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{idx}. {doc_info['Document']}**")
                
                with col2:
                    st.metric("Avg Similarity", f"{doc_info['Avg Similarity'] * 100:.1f}%")
                
                # Progress bar
                progress_val = doc_info['Avg Similarity']
                st.progress(progress_val, text=f"Similarity Score: {progress_val * 100:.1f}%")
    
    with rank_tab2:
        st.markdown("**Documents with lowest average similarity to others**")
        st.caption("These documents are most unique in your collection")
        
        sorted_unique = sorted(doc_avg_similarities, key=lambda x: x['Avg Similarity'])
        
        for idx, doc_info in enumerate(sorted_unique[:10], 1):
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{idx}. {doc_info['Document']}**")
                
                with col2:
                    st.metric("Avg Similarity", f"{doc_info['Avg Similarity'] * 100:.1f}%")
                
                # Progress bar
                progress_val = doc_info['Avg Similarity']
                st.progress(progress_val, text=f"Uniqueness: {(1 - progress_val) * 100:.1f}%")

with analysis_tab3:
    st.markdown("### Network Analysis")
    
    # Threshold for network display
    network_threshold = st.slider(
        "Connection Threshold (%)",
        min_value=50,
        max_value=100,
        value=70,
        help="Only show connections above this similarity"
    ) / 100
    
    # Build network
    import networkx as nx
    
    G = nx.Graph()
    
    # Add nodes
    for idx, doc in enumerate(st.session_state.documents):
        G.add_node(idx, label=doc['name'])
    
    # Add edges
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            similarity = st.session_state.similarity_matrix[i, j]
            if similarity >= network_threshold:
                G.add_edge(i, j, weight=similarity)
    
    # Network statistics
    st.markdown("#### Network Statistics")
    
    net_col1, net_col2, net_col3, net_col4 = st.columns(4)
    
    net_col1.metric("Nodes (Documents)", G.number_of_nodes())
    net_col2.metric("Connections", G.number_of_edges())
    net_col3.metric("Connected Components", nx.number_connected_components(G))
    net_col4.metric("Isolated Documents", sum(1 for node in G.nodes() if G.degree(node) == 0))
    
    # Visualization
    if G.number_of_edges() > 0:
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
                    width=weight * 3,
                    color=f'rgba(100, 100, 100, {weight})'
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
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(st.session_state.documents[node]['name'])
            node_size.append(15 + G.degree(node) * 3)
            node_color.append(G.degree(node))
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Connections"),
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=9),
            hoverinfo='text'
        )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=f"Document Network (Threshold: {network_threshold * 100:.0f}%)",
            showlegend=False,
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No connections found at {network_threshold * 100:.0f}% threshold. Try lowering the threshold.")

with analysis_tab4:
    st.markdown("### Complete Similarity Matrix")
    
    # Display options
    display_mode = st.radio(
        "Display Mode",
        ["Percentage", "Heatmap", "Raw Values"],
        horizontal=True
    )
    
    if display_mode == "Heatmap":
        # Heatmap visualization
        fig = go.Figure(data=go.Heatmap(
            z=st.session_state.similarity_matrix * 100,
            x=[doc['name'][:20] for doc in st.session_state.documents],
            y=[doc['name'][:20] for doc in st.session_state.documents],
            colorscale='RdYlGn',
            text=np.round(st.session_state.similarity_matrix * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 8},
            colorbar=dict(title="Similarity %")
        ))
        
        fig.update_layout(
            title="Full Similarity Matrix",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Table display
        df_matrix = pd.DataFrame(
            st.session_state.similarity_matrix * (100 if display_mode == "Percentage" else 1),
            columns=[doc['name'] for doc in st.session_state.documents],
            index=[doc['name'] for doc in st.session_state.documents]
        )
        
        if display_mode == "Percentage":
            st.dataframe(
                df_matrix.style.format("{:.1f}%").background_gradient(cmap='RdYlGn', vmin=0, vmax=100),
                use_container_width=True
            )
        else:
            st.dataframe(
                df_matrix.style.format("{:.4f}").background_gradient(cmap='RdYlGn', vmin=0, vmax=1),
                use_container_width=True
            )

# Export comprehensive report
st.divider()
st.subheader("💾 Export Comprehensive Report")

export_col1, export_col2 = st.columns(2)

with export_col1:
    # Summary report
    summary_data = {
        'Metric': [
            'Total Documents',
            'Total Comparisons',
            'Average Similarity (%)',
            'Maximum Similarity (%)',
            'Minimum Similarity (%)',
            'Standard Deviation (%)',
            'Very High Similarity Pairs',
            'High Similarity Pairs',
            'Moderate Similarity Pairs',
            'Low Similarity Pairs'
        ],
        'Value': [
            n_docs,
            total_comparisons,
            round(avg_similarity * 100, 2),
            round(max_similarity * 100, 2),
            round(min_similarity * 100, 2),
            round(std_similarity * 100, 2),
            very_high,
            high,
            moderate,
            low
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    csv_summary = df_summary.to_csv(index=False)
    
    st.download_button(
        "📥 Download Summary Report",
        csv_summary,
        "bulk_analysis_summary.csv",
        "text/csv",
        use_container_width=True
    )

with export_col2:
    # Full matrix export
    df_matrix_export = pd.DataFrame(
        st.session_state.similarity_matrix * 100,
        columns=[doc['name'] for doc in st.session_state.documents],
        index=[doc['name'] for doc in st.session_state.documents]
    )
    
    csv_matrix = df_matrix_export.to_csv()
    
    st.download_button(
        "📥 Download Full Matrix",
        csv_matrix,
        "similarity_matrix_full.csv",
        "text/csv",
        use_container_width=True
    )

# Insights and Recommendations
st.divider()
st.subheader("💡 Insights & Recommendations")

insights = []

# Check for duplicates
if very_high > 0:
    insights.append({
        'type': 'warning',
        'title': 'Potential Duplicates Detected',
        'message': f'Found {very_high} document pair(s) with >90% similarity. Review these for duplicate content.'
    })

# Check for diversity
if avg_similarity < 0.3:
    insights.append({
        'type': 'success',
        'title': 'High Content Diversity',
        'message': 'Your document collection is highly diverse. Good for varied content libraries.'
    })
elif avg_similarity > 0.7:
    insights.append({
        'type': 'info',
        'title': 'Low Content Diversity',
        'message': 'Documents are quite similar. Consider adding more diverse content or split into focused sub-collections.'
    })

# Check for outliers
isolated_docs = [doc for doc in doc_avg_similarities if doc['Avg Similarity'] < 0.3]
if isolated_docs:
    insights.append({
        'type': 'info',
        'title': 'Outlier Documents Detected',
        'message': f'Found {len(isolated_docs)} document(s) with low similarity to others. These might be off-topic or unique content.'
    })

# Check collection size
if n_docs < 5:
    insights.append({
        'type': 'info',
        'title': 'Small Collection',
        'message': 'Add more documents for better analysis and organization capabilities.'
    })

# Display insights
if insights:
    for insight in insights:
        if insight['type'] == 'warning':
            st.warning(f"**⚠️ {insight['title']}**: {insight['message']}")
        elif insight['type'] == 'success':
            st.success(f"**✅ {insight['title']}**: {insight['message']}")
        else:
            st.info(f"**ℹ️ {insight['title']}**: {insight['message']}")
else:
    st.success("✅ **All looks good!** Your document collection is well-balanced.")