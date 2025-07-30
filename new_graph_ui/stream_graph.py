import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Workflow Manager",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8080/api"

def get_status_emoji(status):
    status_map = {
        'completed': '✅',
        'running': '🔄',
        'paused': '⏸️',
        'pending': '⏳',
        'error': '❌'
    }
    return status_map.get(status, '❓')

def get_status_color(status):
    color_map = {
        'completed': 'green',
        'running': 'blue',
        'paused': 'orange',
        'pending': 'gray',
        'error': 'red'
    }
    return color_map.get(status, 'gray')

def create_workflow_diagram(workflow):
    if not workflow:
        return go.Figure()
    
    stages = workflow['stages']
    x_positions = list(range(len(stages)))
    y_positions = [0] * len(stages)
    
    fig = go.Figure()
    
    # Add edges
    for i in range(len(stages) - 1):
        fig.add_trace(go.Scatter(
            x=[x_positions[i] + 0.3, x_positions[i+1] - 0.3], 
            y=[y_positions[i], y_positions[i+1]],
            mode='lines',
            line=dict(color='#E5E7EB', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_annotation(
            x=x_positions[i] + 0.65,
            y=0,
            ax=x_positions[i] + 0.35,
            ay=0,
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='#9CA3AF',
            showarrow=True
        )
    
    # Add nodes
    for i, stage in enumerate(stages):
        color = get_status_color(stage['status'])
        size = 60 if stage['status'] == 'running' else 50
        
        fig.add_trace(go.Scatter(
            x=[x_positions[i]], 
            y=[y_positions[i]],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                line=dict(width=3, color='white'),
                opacity=0.9 if stage['status'] == 'running' else 0.7
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x_positions[i]], 
            y=[y_positions[i]],
            mode='text',
            text=[get_status_emoji(stage['status'])],
            textfont=dict(size=24),
            showlegend=False,
            hovertemplate=f"<b>{stage['name']}</b><br>Status: {stage['status'].title()}<br>Duration: {stage['duration']} min<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=[x_positions[i]], 
            y=[-0.6],
            mode='text',
            text=[stage['name']],
            textfont=dict(size=14, color='#374151'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        if stage['duration'] > 0 or stage['status'] == 'running':
            duration_text = f"{stage['duration']} min" if stage['duration'] > 0 else "Running..."
            fig.add_trace(go.Scatter(
                x=[x_positions[i]], 
                y=[0.4],
                mode='text',
                text=[duration_text],
                textfont=dict(size=10, color='#6B7280'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{workflow['name']}</b>",
            font=dict(size=24),
            x=0.5
        ),
        showlegend=False,
        height=400,
        margin=dict(l=50, r=50, t=80, b=100),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-0.5, len(stages) - 0.5]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-1, 1]
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Main Streamlit UI
st.title("🔄 Workflow Management Dashboard")

# API Information
with st.expander("🔗 REST API Information", expanded=False):
    st.markdown(f"""
    **API Base URL:** `{API_BASE_URL}`
    
    **Available Endpoints:**
    - `GET /workflows` - Get all workflows
    - `GET /workflows/{{id}}` - Get specific workflow
    - `POST /workflows` - Create new workflow
    - `PUT /workflows/{{id}}` - Update workflow
    - `DELETE /workflows/{{id}}` - Delete workflow
    - `PUT /workflows/{{id}}/stages/{{stage_id}}/status` - Update stage status
    - `GET /health` - Health check
    
    **Example API Calls:**
    ```bash
    # Get all workflows
    curl {API_BASE_URL}/workflows
    
    # Update stage status
    curl -X PUT {API_BASE_URL}/workflows/{{workflow_id}}/stages/2/status \
      -H "Content-Type: application/json" \
      -d '{{"status": "completed", "duration": 45}}'
    
    # Create new workflow
    curl -X POST {API_BASE_URL}/workflows \
      -H "Content-Type: application/json" \
      -d '{{
        "name": "New Workflow",
        "status": "pending",
        "current_stage": 1,
        "stages": [
          {{"id": 1, "name": "Stage 1", "status": "pending", "duration": 0}}
        ]
      }}'
    ```
    """)

# Fetch workflows
try:
    response = requests.get(f"{API_BASE_URL}/workflows")
    response.raise_for_status()
    workflows = response.json()
except requests.RequestException as e:
    st.error(f"Failed to fetch workflows: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🔄 Workflow Selector")
    
    if not workflows:
        st.warning("No workflows available")
        st.stop()
    
    # Workflow selection
    workflow_options = list(workflows.keys())
    if 'selected_workflow' not in st.session_state or st.session_state.selected_workflow not in workflow_options:
        st.session_state.selected_workflow = workflow_options[0]
    
    selected_workflow = st.selectbox(
        "Choose a workflow:",
        options=workflow_options,
        format_func=lambda x: f"{get_status_emoji(workflows[x]['status'])} {workflows[x]['name']}",
        key="workflow_selector",
        index=workflow_options.index(st.session_state.selected_workflow) if st.session_state.selected_workflow in workflow_options else 0
    )
    
    st.session_state.selected_workflow = selected_workflow
    
    st.markdown("---")
    
    # Workflow info
    if selected_workflow:
        workflow = workflows[selected_workflow]
        st.subheader("Workflow Info")
        st.write(f"**ID:** `{workflow['id'][:8]}...`")
        st.write(f"**Status:** {get_status_emoji(workflow['status'])} {workflow['status'].title()}")
        st.write(f"**Current Stage:** {workflow['current_stage']}/{len(workflow['stages'])}")
        st.write(f"**Total Stages:** {len(workflow['stages'])}")
        st.write(f"**Last Updated:** {datetime.fromisoformat(workflow['updated']).strftime('%H:%M:%S')}")
        
        completed_stages = sum(1 for stage in workflow['stages'] if stage['status'] == 'completed')
        progress = (completed_stages / len(workflow['stages'])) * 100
        st.progress(progress / 100)
        st.write(f"**Progress:** {progress:.0f}%")
        
        st.markdown("---")
        
        # Workflow controls
        st.subheader("Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                try:
                    response = requests.put(
                        f"{API_BASE_URL}/workflows/{selected_workflow}",
                        json={'status': 'running'}
                    )
                    response.raise_for_status()
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Failed to start workflow: {str(e)}")
        
        with col2:
            if st.button("⏸️ Pause", use_container_width=True):
                try:
                    response = requests.put(
                        f"{API_BASE_URL}/workflows/{selected_workflow}",
                        json={'status': 'paused'}
                    )
                    response.raise_for_status()
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Failed to pause workflow: {str(e)}")

# Main content area
if st.session_state.selected_workflow:
    workflow = workflows[st.session_state.selected_workflow]
    
    st.header(f"Current Workflow: {workflow['name']}")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", f"{get_status_emoji(workflow['status'])} {workflow['status'].title()}")
    with col2:
        current_stage_name = workflow['stages'][workflow['current_stage']-1]['name']
        st.metric("Current Stage", f"{workflow['current_stage']}: {current_stage_name}")
    with col3:
        completed = sum(1 for stage in workflow['stages'] if stage['status'] == 'completed')
        st.metric("Completed", f"{completed}/{len(workflow['stages'])}")
    with col4:
        total_duration = sum(stage['duration'] for stage in workflow['stages'])
        st.metric("Duration", f"{total_duration} min")
    
    # Workflow diagram
    fig = create_workflow_diagram(workflow)
    st.plotly_chart(fig, use_container_width=True)
    
    # Stage details
    st.subheader("Stage Details")
    stage_data = []
    for i, stage in enumerate(workflow['stages'], 1):
        started = datetime.fromisoformat(stage['started_at']).strftime('%H:%M:%S') if stage.get('started_at') else 'Not started'
        completed = datetime.fromisoformat(stage['completed_at']).strftime('%H:%M:%S') if stage.get('completed_at') else 'Not completed'
        stage_data.append({
            'Stage': str(i),
            'Name': str(stage['name']),
            'Status': f"{get_status_emoji(stage['status'])} {stage['status'].title()}",
            'Duration (min)': str(stage['duration']),
            'Started': started,
            'Completed': completed
        })
    
    df = pd.DataFrame(stage_data)
    for col in df.columns:
        df[col] = df[col].astype(str)
    st.dataframe(df, use_container_width=True, hide_index=True)

# Auto-refresh
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Refresh"):
        st.rerun()

with col2:
    auto_refresh = st.checkbox("Auto-refresh every 5 seconds")

if auto_refresh:
    time.sleep(5)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | API Server: Running on port 8081*")