import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page header
st.title("⚠️ Plagiarism Detection")
st.markdown("### Check Content Originality")

# Check prerequisites
if 'similarity_matrix' not in st.session_state or st.session_state.similarity_matrix is None:
    st.warning("⚠️ No documents analyzed yet!")
    
    if st.button("📤 Go to Upload & Analyze"):
        st.switch_page("upload_analyze.py")
    st.stop()

# Explanation
with st.expander("ℹ️ How Plagiarism Detection Works"):
    st.markdown("""
    ### Detection Method
    
    Our system uses **cosine similarity** to compare documents:
    - **90-100%**: Very High Risk - Nearly identical content
    - **75-90%**: High Risk - Significant overlap
    - **60-75%**: Moderate Risk - Notable similarities
    - **<60%**: Low Risk - Minimal overlap
    
    ### Best Practices
    
    1. **This is a screening tool**, not definitive proof
    2. Always manually review flagged documents
    3. Consider context and citation practices
    4. Use multiple plagiarism detection tools
    5. Check for proper attributions
    
    ### Limitations
    
    - Cannot detect paraphrasing without citation
    - Doesn't check against external databases
    - May flag legitimate quotes or common phrases
    - Works best with substantial text samples
    """)

st.divider()

# Configuration
st.subheader("⚙️ Detection Settings")

col1, col2 = st.columns(2)

with col1:
    risk_threshold = st.select_slider(
        "Risk Threshold",
        options=["Lenient (75%)", "Moderate (70%)", "Strict (60%)", "Very Strict (50%)"],
        value="Moderate (70%)",
        help="How strict should the plagiarism detection be?"
    )
    
    # Extract percentage
    threshold_value = int(risk_threshold.split('(')[1].split('%')[0]) / 100

with col2:
    check_mode = st.radio(
        "Check Mode",
        ["Check Against All Documents", "Check Specific Document"],
        help="Choose whether to check all documents or a specific one"
    )

st.divider()

# Analysis based on mode
if check_mode == "Check Specific Document":
    st.subheader("🎯 Select Document to Check")
    
    selected_doc_idx = st.selectbox(
        "Document to analyze",
        options=range(len(st.session_state.documents)),
        format_func=lambda x: st.session_state.documents[x]['name']
    )
    
    st.divider()
    
    # Analyze specific document
    doc_similarities = []
    
    for i in range(len(st.session_state.documents)):
        if i != selected_doc_idx:
            similarity = st.session_state.similarity_matrix[selected_doc_idx, i]
            
            if similarity > 0.9:
                risk = "Very High"
                color = "🔴"
            elif similarity > 0.75:
                risk = "High"
                color = "🟠"
            elif similarity > 0.6:
                risk = "Moderate"
                color = "🟡"
            else:
                risk = "Low"
                color = "🟢"
            
            doc_similarities.append({
                'Document': st.session_state.documents[i]['name'],
                'Similarity': similarity * 100,
                'Risk': risk,
                'Color': color,
                'Index': i
            })
    
    # Sort by similarity
    doc_similarities.sort(key=lambda x: x['Similarity'], reverse=True)
    
    # Display results
    st.subheader(f"📊 Checking: {st.session_state.documents[selected_doc_idx]['name']}")
    
    # Risk summary
    very_high_risk = sum(1 for d in doc_similarities if d['Risk'] == 'Very High')
    high_risk = sum(1 for d in doc_similarities if d['Risk'] == 'High')
    moderate_risk = sum(1 for d in doc_similarities if d['Risk'] == 'Moderate')
    low_risk = sum(1 for d in doc_similarities if d['Risk'] == 'Low')
    
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    
    risk_col1.metric("🔴 Very High", very_high_risk)
    risk_col2.metric("🟠 High", high_risk)
    risk_col3.metric("🟡 Moderate", moderate_risk)
    risk_col4.metric("🟢 Low", low_risk)
    
    # Overall verdict
    if very_high_risk > 0:
        st.error(f"""
        ### ⚠️ PLAGIARISM ALERT
        
        {very_high_risk} document(s) show very high similarity (>90%).
        **Immediate review required!**
        """)
    elif high_risk > 0:
        st.warning(f"""
        ### ⚠️ High Risk Detected
        
        {high_risk} document(s) show high similarity (75-90%).
        Manual review recommended.
        """)
    else:
        st.success("""
        ### ✅ Low Risk
        
        No significant plagiarism detected.
        Document appears to be original content.
        """)
    
    st.divider()
    
    # Detailed results
    st.markdown("#### 📋 Detailed Comparison Results")
    
    # Show only documents above threshold
    flagged_docs = [d for d in doc_similarities if d['Similarity'] >= threshold_value * 100]
    
    if flagged_docs:
        for doc_info in flagged_docs:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### {doc_info['Color']} {doc_info['Document']}")
                
                with col2:
                    st.metric("Similarity", f"{doc_info['Similarity']:.1f}%")
                    st.caption(f"{doc_info['Risk']} Risk")
                
                with st.expander("🔍 View Comparison"):
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown(f"**Target Document**")
                        st.markdown(f"*{st.session_state.documents[selected_doc_idx]['name']}*")
                        content1 = st.session_state.documents[selected_doc_idx]['content']
                        st.text_area(
                            "Content",
                            content1[:400] + "..." if len(content1) > 400 else content1,
                            height=200,
                            disabled=True,
                            key=f"plag_target_{doc_info['Index']}",
                            label_visibility="collapsed"
                        )
                    
                    with comp_col2:
                        st.markdown(f"**Matching Document**")
                        st.markdown(f"*{doc_info['Document']}*")
                        content2 = st.session_state.documents[doc_info['Index']]['content']
                        st.text_area(
                            "Content",
                            content2[:400] + "..." if len(content2) > 400 else content2,
                            height=200,
                            disabled=True,
                            key=f"plag_match_{doc_info['Index']}",
                            label_visibility="collapsed"
                        )
                    
                    # Common terms
                    doc1_processed = st.session_state.processed_documents[selected_doc_idx]['processed']
                    doc2_processed = st.session_state.processed_documents[doc_info['Index']]['processed']
                    
                    common_words = set(doc1_processed.split()) & set(doc2_processed.split())
                    
                    if common_words:
                        st.markdown("**🔤 Common Terms:**")
                        st.caption(", ".join(sorted(list(common_words))[:40]))
    else:
        st.info(f"No documents found with similarity ≥ {threshold_value * 100}%")
    
    # Visualization
    st.divider()
    st.markdown("#### 📊 Similarity Distribution")
    
    fig = go.Figure(data=[
        go.Bar(
            x=[d['Document'] for d in doc_similarities],
            y=[d['Similarity'] for d in doc_similarities],
            marker_color=['red' if d['Risk'] == 'Very High' else 
                         'orange' if d['Risk'] == 'High' else
                         'yellow' if d['Risk'] == 'Moderate' else
                         'green' for d in doc_similarities]
        )
    ])
    
    fig.update_layout(
        title=f"Similarity Scores for: {st.session_state.documents[selected_doc_idx]['name']}",
        xaxis_title="Compared Documents",
        yaxis_title="Similarity (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:  # Check Against All Documents
    st.subheader("📊 Full Plagiarism Scan")
    
    # Find all high-similarity pairs
    plagiarism_cases = []
    n_docs = len(st.session_state.documents)
    
    # Ensure similarity matrix matches document count
    if st.session_state.similarity_matrix.shape[0] != n_docs:
        st.error("⚠️ Similarity matrix doesn't match document count. Please re-analyze documents.")
        if st.button("🔄 Go to Upload & Analyze"):
            st.switch_page("upload_analyze.py")
        st.stop()
    
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            similarity = st.session_state.similarity_matrix[i, j]
            
            if similarity >= threshold_value:
                if similarity > 0.9:
                    risk = "Very High"
                elif similarity > 0.75:
                    risk = "High"
                elif similarity > 0.6:
                    risk = "Moderate"
                else:
                    risk = "Low"
                
                plagiarism_cases.append({
                    'Doc1_Index': i,
                    'Doc1': st.session_state.documents[i]['name'],
                    'Doc2_Index': j,
                    'Doc2': st.session_state.documents[j]['name'],
                    'Similarity': similarity * 100,
                    'Risk': risk
                })
    
    # Sort by risk and similarity
    risk_order = {'Very High': 0, 'High': 1, 'Moderate': 2, 'Low': 3}
    plagiarism_cases.sort(key=lambda x: (risk_order[x['Risk']], -x['Similarity']))
    
    # Summary
    very_high = sum(1 for c in plagiarism_cases if c['Risk'] == 'Very High')
    high = sum(1 for c in plagiarism_cases if c['Risk'] == 'High')
    moderate = sum(1 for c in plagiarism_cases if c['Risk'] == 'Moderate')
    low = sum(1 for c in plagiarism_cases if c['Risk'] == 'Low')
    
    sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)
    
    sum_col1.metric("Total Cases", len(plagiarism_cases))
    sum_col2.metric("🔴 Very High", very_high)
    sum_col3.metric("🟠 High", high)
    sum_col4.metric("🟡 Moderate", moderate)
    sum_col5.metric("🟢 Low", low)
    
    # Overall verdict
    if very_high > 0:
        st.error(f"""
        ### ⚠️ CRITICAL: Plagiarism Detected
        
        Found {very_high} case(s) of very high similarity (>90%).
        **Immediate action required!**
        """)
    elif high > 0:
        st.warning(f"""
        ### ⚠️ Warning: High Similarity Detected
        
        Found {high} case(s) of high similarity (75-90%).
        Review recommended.
        """)
    elif moderate > 0:
        st.info(f"""
        ### ℹ️ Moderate Similarity Found
        
        Found {moderate} case(s) of moderate similarity (60-75%).
        May warrant investigation.
        """)
    else:
        st.success("""
        ### ✅ All Clear
        
        No significant plagiarism detected across all documents.
        """)
    
    st.divider()
    
    # Display cases
    if plagiarism_cases:
        st.markdown("#### 📋 Detected Cases")
        
        for idx, case in enumerate(plagiarism_cases):
            color = "🔴" if case['Risk'] == 'Very High' else "🟠" if case['Risk'] == 'High' else "🟡" if case['Risk'] == 'Moderate' else "🟢"
            
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### {color} Case #{idx + 1}")
                    st.markdown(f"**📄 {case['Doc1']}**")
                    st.markdown(f"**📄 {case['Doc2']}**")
                
                with col2:
                    st.metric("Similarity", f"{case['Similarity']:.1f}%")
                    st.caption(f"{case['Risk']} Risk")
                
                with st.expander("🔍 View Details"):
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown(f"**{case['Doc1']}**")
                        content1 = st.session_state.documents[case['Doc1_Index']]['content']
                        st.text_area(
                            "Content",
                            content1[:400] + "..." if len(content1) > 400 else content1,
                            height=200,
                            disabled=True,
                            key=f"plag_case1_{idx}",
                            label_visibility="collapsed"
                        )
                    
                    with detail_col2:
                        st.markdown(f"**{case['Doc2']}**")
                        content2 = st.session_state.documents[case['Doc2_Index']]['content']
                        st.text_area(
                            "Content",
                            content2[:400] + "..." if len(content2) > 400 else content2,
                            height=200,
                            disabled=True,
                            key=f"plag_case2_{idx}",
                            label_visibility="collapsed"
                        )
        
        # Distribution chart
        st.divider()
        st.markdown("#### 📊 Risk Distribution")
        
        risk_counts = {'Very High': very_high, 'High': high, 'Moderate': moderate, 'Low': low}
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                marker_colors=['#ff4444', '#ffaa00', '#ffff00', '#44ff44']
            )
        ])
        
        fig.update_layout(
            title="Cases by Risk Level",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info(f"No plagiarism cases found at threshold ≥ {threshold_value * 100}%")

# Export
st.divider()
st.subheader("💾 Export Report")

if check_mode == "Check Specific Document":
    if flagged_docs:
        df_export = pd.DataFrame([
            {
                'Target Document': st.session_state.documents[selected_doc_idx]['name'],
                'Matching Document': d['Document'],
                'Similarity (%)': round(d['Similarity'], 2),
                'Risk Level': d['Risk']
            }
            for d in flagged_docs
        ])
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            "📥 Download Plagiarism Report (CSV)",
            csv,
            "plagiarism_report.csv",
            "text/csv",
            use_container_width=True
        )
else:
    if plagiarism_cases:
        df_export = pd.DataFrame([
            {
                'Document 1': case['Doc1'],
                'Document 2': case['Doc2'],
                'Similarity (%)': round(case['Similarity'], 2),
                'Risk Level': case['Risk']
            }
            for case in plagiarism_cases
        ])
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            "📥 Download Full Report (CSV)",
            csv,
            "plagiarism_scan_report.csv",
            "text/csv",
            use_container_width=True
        )

# Recommendations
st.divider()
st.subheader("💡 Next Steps")

if very_high > 0 or high > 0:
    st.markdown("""
    ### Recommended Actions:
    
    1. **Manual Review**: Carefully examine flagged documents
    2. **Check Citations**: Verify if content is properly attributed
    3. **Compare Side-by-Side**: Look for paraphrasing vs. direct copying
    4. **Contact Authors**: If student submissions, discuss with students
    5. **Use Additional Tools**: Cross-check with other plagiarism detectors
    6. **Document Findings**: Keep records of investigation
    """)
else:
    st.markdown("""
    ### All Clear!
    
    Your documents appear to be original. Continue with:
    - 🗂️ Auto-organizing documents by topic
    - 📊 Analyzing document relationships
    - 🔍 Finding similar content for reference
    """)