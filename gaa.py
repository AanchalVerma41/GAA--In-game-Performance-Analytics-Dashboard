import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="GAA Match Analysis Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme-Agnostic CSS for Borders ---
st.markdown("""
<style>
    .chart-container {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .event-feed p { margin-bottom: 5px !important; font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# --- Data Loading and Processing ---
@st.cache_data
def load_and_process_data(path):
    df = pd.read_csv(path, engine='python')

    # Data Cleaning
    score_cols = ['Cork Goals', 'Cork Points', 'Galway Goals', 'Galway Points']
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Is_Turnover'] = (df['Turnover?'] == 'Yes').astype(int)
    df['Is_Shot_Attempt'] = (df['Shot Attempt?'] == 'Yes').astype(int)
    df['Possession_Start'] = (df['Possession Start?'] == 'Yes').astype(int)
    df['Shot_Outcome_Std'] = np.select([df['Shot Outcome'].isin(['Goal']), df['Shot Outcome'].isin(['Point', 'Point over the bar', 'Successful'])], ['Goal', 'Point'], default='Miss')
    df.loc[df['Is_Shot_Attempt'] == 0, 'Shot_Outcome_Std'] = 'No Shot'
    df['Is_Score'] = (df['Shot_Outcome_Std'].isin(['Goal', 'Point'])).astype(int)
    df['Notes'] = df['Notes'].fillna('')
    df = df[df['Timestamp'].str.strip() != 'Half Time'].copy()

    # Timeline Creation
    def to_total_seconds(t_str):
        if pd.isna(t_str): return 0
        try:
            parts = str(t_str).split(':'); h, m, s = (map(int, parts) if len(parts) == 3 else (0, *map(int, parts))); return h * 3600 + m * 60 + s
        except: return 0
    df['TimeLeftSeconds'] = df['CleanedTImestamp'].apply(to_total_seconds)
    half_time_idx = (df['TimeLeftSeconds'].diff() < -1000).idxmax()
    df['Half'] = np.where(df.index >= half_time_idx, 2, 1) if half_time_idx > 0 else 1
    df['GameSeconds'] = df.groupby('Half')['TimeLeftSeconds'].transform(lambda x: x.max() - x)
    if 2 in df['Half'].values:
        first_half_dur = df.loc[df['Half'] == 1, 'GameSeconds'].max()
        df.loc[df['Half'] == 2, 'GameSeconds'] += first_half_dur
    df = df.sort_values(by='GameSeconds').reset_index(drop=True)

    # --- NORMALIZE TIMELINE to 60 minutes (3600 seconds) ---
    actual_max_time = df['GameSeconds'].max()
    target_max_time = 3600
    if actual_max_time > 0:
        scaling_factor = target_max_time / actual_max_time
        df['GameSeconds'] = df['GameSeconds'] * scaling_factor

    # Score, Stat, and Advanced Metric Calculation
    df['Cork_Event_Score'] = df['Cork Goals'] * 3 + df['Cork Points']
    df['Galway_Event_Score'] = df['Galway Goals'] * 3 + df['Galway Points']
    df['Cork_Total_Score'] = df['Cork_Event_Score'].cumsum()
    df['Galway_Total_Score'] = df['Galway_Event_Score'].cumsum()
    
    df['Cork_Total_Goals'] = df['Cork Goals'].cumsum()
    df['Cork_Total_Points'] = df['Cork Points'].cumsum()
    df['Galway_Total_Goals'] = df['Galway Goals'].cumsum()
    df['Galway_Total_Points'] = df['Galway Points'].cumsum()
    
    df['Score_Difference'] = df['Cork_Total_Score'] - df['Galway_Total_Score']
    df['Cork_Possession_Count'] = ((df['Team'] == 'Cork') & (df['Possession_Start'] == 1)).cumsum()
    df['Galway_Possession_Count'] = ((df['Team'] == 'Galway') & (df['Possession_Start'] == 1)).cumsum()
    df['Cork_Productivity'] = (df['Cork_Total_Score'] / df['Cork_Possession_Count']).fillna(0)
    df['Galway_Productivity'] = (df['Galway_Total_Score'] / df['Galway_Possession_Count']).fillna(0)
    df['Cork_Shot_Flag'] = ((df['Team'] == 'Cork') & (df['Is_Shot_Attempt'] == 1)).astype(int)
    df['Galway_Shot_Flag'] = ((df['Team'] == 'Galway') & (df['Is_Shot_Attempt'] == 1)).astype(int)
    df['Cork_Forced_Turnover'] = ((df['Team'] == 'Galway') & (df['Is_Turnover'] == 1)).astype(int)
    df['Galway_Forced_Turnover'] = ((df['Team'] == 'Cork') & (df['Is_Turnover'] == 1)).astype(int)
    df_timed = df.set_index(pd.to_timedelta(df['GameSeconds'], unit='s'))
    cork_pressure_series = (df_timed['Cork_Shot_Flag'].rolling('180s').sum()) + (df_timed['Cork_Forced_Turnover'].rolling('180s').sum() * 1.5)
    galway_pressure_series = (df_timed['Galway_Shot_Flag'].rolling('180s').sum()) + (df_timed['Galway_Forced_Turnover'].rolling('180s').sum() * 1.5)
    df['Cork_Pressure'] = cork_pressure_series.values
    df['Galway_Pressure'] = galway_pressure_series.values
    df[['Cork_Pressure', 'Galway_Pressure']] = df[['Cork_Pressure', 'Galway_Pressure']].interpolate().fillna(0)
    df['Pressure_Index'] = df['Cork_Pressure'] - df['Galway_Pressure']
    df['Possession_ID'] = df['Possession_Start'].cumsum()
    possession_outcomes = df.groupby('Possession_ID').agg(Team=('Team', 'first'), Points=('Cork_Event_Score', 'sum'), GameSeconds=('GameSeconds', 'last'))
    possession_outcomes.loc[possession_outcomes['Team'] == 'Galway', 'Points'] = df.groupby('Possession_ID')['Galway_Event_Score'].sum()
    cork_epp = possession_outcomes[possession_outcomes['Team']=='Cork']['Points'].rolling(window=10, min_periods=1).mean()
    galway_epp = possession_outcomes[possession_outcomes['Team']=='Galway']['Points'].rolling(window=10, min_periods=1).mean()
    possession_outcomes['Cork_EPP'] = cork_epp
    possession_outcomes['Galway_EPP'] = galway_epp
    possession_outcomes.fillna(method='ffill', inplace=True)
    df = pd.merge(df, possession_outcomes[['GameSeconds', 'Cork_EPP', 'Galway_EPP']], on='GameSeconds', how='left').fillna(method='ffill').fillna(0)
    max_time_val = df['GameSeconds'].max()
    df['Time_Remaining_Factor'] = ((max_time_val - df['GameSeconds']) / max_time_val) ** 0.75 if max_time_val > 0 else 0
    win_score = (df['Score_Difference'] / (df['Time_Remaining_Factor'] + 0.1))
    df['Cork_WP'] = 1 / (1 + np.exp(-win_score * 0.1))
    
    return df

# --- Main App Logic ---
try:
    df = load_and_process_data('gaa.csv')

    # --- Sidebar ---
    st.sidebar.title("Match Controls & Events")
    st.sidebar.markdown("---")
    st.sidebar.header("Game Timeline")
    max_time = int(df['GameSeconds'].max())
    current_time = st.sidebar.slider("Game Time", 0, max_time, max_time, 1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Key Events Feed")
    events_df = df[df['GameSeconds'] <= current_time]
    events_with_notes = events_df[events_df['Notes'] != ''].sort_values(by='GameSeconds', ascending=False)
    for _, row in events_with_notes.iterrows():
        event_time_str = time.strftime('%M:%S', time.gmtime(row['GameSeconds']))
        note_text = row['Notes']
        if row['Is_Score'] == 1:
            color = 'red' if row['Team'] == 'Cork' else 'purple'
            st.sidebar.markdown(f"<p style='color:{color}; margin-bottom: 2px;'><b>{event_time_str}</b> - {note_text}</p>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"<p style='margin-bottom: 2px;'><b>{event_time_str}</b> - {note_text}</p>", unsafe_allow_html=True)

    filtered_df = df[df['GameSeconds'] <= current_time].copy()
    
    # --- Main Dashboard ---
    st.title("Ladies Gaelic Football Match Cork vs Galway 2024")
    st.header("Scoreboard")
    
    if not filtered_df.empty:
        last_row = filtered_df.iloc[-1]
        cork_g, cork_p, cork_total = int(last_row['Cork_Total_Goals']), int(last_row['Cork_Total_Points']), int(last_row['Cork_Total_Score'])
        galway_g, galway_p, galway_total = int(last_row['Galway_Total_Goals']), int(last_row['Galway_Total_Points']), int(last_row['Galway_Total_Score'])
    else:
        cork_g, cork_p, cork_total = 0, 0, 0
        galway_g, galway_p, galway_total = 0, 0, 0

    cork_display_score = f"{cork_g}-{cork_p} ({cork_total})"
    galway_display_score = f"{galway_g}-{galway_p} ({galway_total})"
    
    col1, col2 = st.columns(2)
    col1.metric(label="🔴 Cork", value=cork_display_score)
    col2.metric(label="🟣 Galway", value=galway_display_score)

    st.markdown("---")

    # --- Charting Helper for Minute Labels ---
    max_seconds = filtered_df['GameSeconds'].max() if not filtered_df.empty else 3600
    tick_interval_seconds = 300 # 5 minutes
    tick_values = np.arange(0, max_seconds + tick_interval_seconds, tick_interval_seconds)
    tick_labels = [f'{int(s/60)}m' for s in tick_values]

    # --- Tabbed Layout for Charts ---
    tab1, tab2, tab3 = st.tabs(["📊 Match Overview", "🚀 Performance Analysis", "🧠 Advanced Analytics"])

    with tab1:
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Total Shots")
            fig = go.Figure()
            for team, color in [('Cork', 'red'), ('Galway', 'purple')]:
                shots = filtered_df[filtered_df['Team'] == team].groupby('GameSeconds')['Is_Shot_Attempt'].sum().cumsum()
                fig.add_trace(go.Scatter(x=shots.index, y=shots.values, mode='lines', name=team, line=dict(color=color)))
            fig.update_layout(xaxis_title='Game Time (minutes)', margin=dict(t=30, b=0), xaxis=dict(tickmode='array', tickvals=tick_values, ticktext=tick_labels))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Total Turnovers")
            fig = go.Figure()
            for team, color in [('Cork', 'red'), ('Galway', 'purple')]:
                turnovers = filtered_df[filtered_df['Team'] == team].groupby('GameSeconds')['Is_Turnover'].sum().cumsum()
                fig.add_trace(go.Scatter(x=turnovers.index, y=turnovers.values, mode='lines', name=team, line=dict(color=color)))
            fig.update_layout(xaxis_title='Game Time (minutes)', margin=dict(t=30, b=0), xaxis=dict(tickmode='array', tickvals=tick_values, ticktext=tick_labels))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Score Progression & Shot Outcomes")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=filtered_df['Score_Difference'], mode='lines', line=dict(color='grey', width=2), name='Score Difference'))
        shot_df = filtered_df[filtered_df['Is_Shot_Attempt'] == 1].copy()
        for team, color in [('Cork', 'red'), ('Galway', 'purple')]:
            team_shots = shot_df[shot_df['Team'] == team]
            fig.add_trace(go.Scatter(x=team_shots['GameSeconds'], y=team_shots['Score_Difference'], mode='markers', name=team, marker_symbol=team_shots['Shot_Outcome_Std'].map({'Goal': 'star', 'Point': 'circle', 'Miss': 'x-thin'}), marker=dict(color=color, size=11, line=dict(width=1, color='black'))))
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
        fig.update_layout(xaxis_title='Game Time (minutes)', yaxis_title='Score Difference (Cork - Galway)', margin=dict(t=30, b=0), xaxis=dict(tickmode='array', tickvals=tick_values, ticktext=tick_labels))
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        col5, col6 = st.columns(2)
        with col5:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Shot Success Rate (%) & Productivity")
            cork_total = filtered_df[filtered_df['Team'] == 'Cork']['Is_Shot_Attempt'].sum()
            galway_total = filtered_df[filtered_df['Team'] == 'Galway']['Is_Shot_Attempt'].sum()
            cork_success_rate = (filtered_df[filtered_df['Team'] == 'Cork']['Is_Score'].sum() / cork_total) * 100 if cork_total > 0 else 0
            galway_success_rate = (filtered_df[filtered_df['Team'] == 'Galway']['Is_Score'].sum() / galway_total) * 100 if galway_total > 0 else 0
            scol1, scol2 = st.columns(2)
            scol1.metric("Cork Success", f"{cork_success_rate:.1f}%")
            scol2.metric("Galway Success", f"{galway_success_rate:.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=filtered_df['Cork_Productivity'], mode='lines', name='Cork', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=filtered_df['Galway_Productivity'], mode='lines', name='Galway', line=dict(color='purple')))
            fig.update_layout(xaxis_title='Game Time (minutes)', yaxis_title='Points Per Possession', margin=dict(t=30, b=0), xaxis=dict(tickmode='array', tickvals=tick_values, ticktext=tick_labels))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Team Pressure Index (3-Min Roll)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=filtered_df['Pressure_Index'], mode='lines', line=dict(color='grey'), name='Pressure'))
            fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
            fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=filtered_df['Pressure_Index'].where(filtered_df['Pressure_Index'] >= 0), mode='lines', fill='tozeroy', line_color='rgba(255,0,0,0)', fillcolor='rgba(255, 71, 71, 0.4)', hoverinfo='none'))
            fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=filtered_df['Pressure_Index'].where(filtered_df['Pressure_Index'] <= 0), mode='lines', fill='tozeroy', line_color='rgba(128,0,128,0)', fillcolor='rgba(218, 112, 214, 0.4)', hoverinfo='none'))
            fig.update_layout(xaxis_title='Game Time (minutes)', yaxis_title='Pressure Index', showlegend=False, margin=dict(t=30, b=0), xaxis=dict(tickmode='array', tickvals=tick_values, ticktext=tick_labels))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        col7, col8 = st.columns(2)
        with col7:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Win Probability")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=filtered_df['Cork_WP'], name='Cork Win Prob.', line=dict(color='red'), fill='tozeroy'))
            fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=1-filtered_df['Cork_WP'], name='Galway Win Prob.', line=dict(color='purple'), fill='tonexty'))
            fig.update_layout(xaxis_title='Game Time (minutes)', yaxis_title='Probability', yaxis_range=[0,1], margin=dict(t=30, b=0), xaxis=dict(tickmode='array', tickvals=tick_values, ticktext=tick_labels))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.markdown('</div>', unsafe_allow_html=True)

        with col8:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Expected Points Per Possession (EPP)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=filtered_df['Cork_EPP'], name='Cork EPP', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=filtered_df['GameSeconds'], y=filtered_df['Galway_EPP'], name='Galway EPP', line=dict(color='purple')))
            fig.update_layout(xaxis_title='Game Time (minutes)', yaxis_title='Expected Points (Rolling 10 Poss.)', margin=dict(t=30, b=0), xaxis=dict(tickmode='array', tickvals=tick_values, ticktext=tick_labels))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")