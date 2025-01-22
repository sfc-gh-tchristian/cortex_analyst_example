# In SiS, import the following packages: plotly, snowflake, snowflake-ml-python, streamlit-extras, snowflake-snowpark-python
import streamlit as st
import _snowflake
import snowflake.cortex as cortex
import json
import math
from streamlit_extras.stylable_container import stylable_container
from snowflake.snowpark.context import get_active_session
import plotly.express as px
from typing import Dict, List, Optional

###############
# APP CONFIG  #
###############

session = get_active_session() # If using outside of Snowflake, provide credentials.

llm = "claude-3-5-sonnet" # Used to generate charts. mistral-large2 is great if outside of the US.

# Define your semantic models below
MODELS = [
    {
        "name": "**Movie Model**", "icon": "🎬", 
        "description": "Wondering what to watch tonight? Why not check out the IMDB dataset.",
        "file": "LLM_DEMO.ANALYST.SEMANTIC/movie_model_lsq.yaml",
        "table": "LLM_DEMO.ANALYST.MOVIE_FULL"
    },
    {
        "name": "**Sales Model**", "icon": "💰",
        "description": "Your classic business example - ask data of fictional sales data.",
        "file": "CORTEX_ANALYST_DEMO.REVENUE_TIMESERIES.RAW_DATA/revenue_timeseries.yaml",
        "table": "CORTEX_ANALYST_DEMO.REVENUE_TIMESERIES.DAILY_REVENUE_BY_PRODUCT"
    },
    {
        "name": "**CRM Model**", "icon": "👨🏻‍💻",
        "description": "CRM data sample from a popular system, contains account, opp, and user data.",
        "file": "SFDC.SALESCLOUD.SEMANTICS/sfdc_sales_demo.yaml",
        "table": "SFDC.SALESCLOUD.OPPORTUNITY"
    },
    {
        "name": "**Whisky Model**", "icon": "🥃",
        "description": "Quench your thirst with the Meta-Critic Whisky Database.",
        "file": "LLM_DEMO.ANALYST.SEMANTIC/whisky.yaml",
        "table": "LLM_DEMO.ANALYST.WHISKY_TABLE"
    }
]

###############
# STYLE SHEET #
###############

vLogo = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Snowflake_Logo.svg/1024px-Snowflake_Logo.svg.png'      
st.markdown("""
    <style>
    div.block-container{margin-top:0; padding-top:0;position:relative;}
    [data-testid=stAppViewContainer] {background-color: #F2F0E9}
    [data-testid=stSidebar] {background-color: #E6E3D8}
    h1, h2, h3, h4, h5, h6, a, p, span, ul, li, ol {color: #2A2725;}
    </style>
""", unsafe_allow_html=True) #HERE ARE THE VISUAL ELEMENTS, IGNORE THIS UNLESS YOU'D LIKE TO ALTER THE STYLE

###############
#  FUNCTIONS  #
###############

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'active_suggestion' not in st.session_state:
        st.session_state.active_suggestion = None

def yaml_interactive_selector():
    st.sidebar.title("Semantic Model Selector")
    
    if st.session_state.selected_model == None:
        st.sidebar.warning('No data domain selected, pick one below!', icon="⚠️")
    
    for model in MODELS:
        with st.sidebar.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                with st.popover(model['icon']):
                    st.dataframe(session.sql(f"SELECT * FROM {model['table']} LIMIT 5"))
            with col2:
                st.write(model['name'])
                st.caption(model['description'])
            
            if st.sidebar.button(f"Select {model['name']}", key=f"model_{model['name']}"):
                st.session_state.selected_model = model['file']

            if st.session_state.selected_model == model['file']:    
                st.sidebar.success(f"Selected: {model['name']}")

def send_message() -> dict:
    """Calls the REST API and returns the response."""
    
    request_body = {
        "messages": st.session_state.messages,
        "semantic_model_file": f"@{st.session_state.selected_model}",
    }
    
    resp = _snowflake.send_snow_api_request(
        "POST",f"/api/v2/cortex/analyst/message",
        {},{},request_body,{},30000,)

    if resp["status"] < 400:
        return json.loads(resp["content"])
    else:
        raise Exception(
            f"Failed request with status {resp['status']}: {resp}"
        )

def process_message(prompt: str) -> None:
    """Processes a message and adds the response to the chat."""
    st.session_state.messages.append(
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = send_message()
            request_id = response["request_id"]
            content = response["message"]["content"]
            st.session_state.messages.append(
                {"role": "analyst", "content": content, "request_id": request_id}
            )
            display_content(content=content, request_id=request_id)  # type: ignore[arg-type]

def millify(n, decimal_places=0):
    MILLNAMES = ['',' K',' M',' B',' T']
    n = float(n)
    millidx = max(0, min(len(MILLNAMES)-1,int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return f"{n / 10**(3 * millidx):.{decimal_places}f}{MILLNAMES[millidx]}"

def get_chart_suggestions(df):
    prompt =  f"""
    Analyze this question, dataframe structure and sample data to suggest visualization parameters using visual best practice (as dictated by the Financial Times guidelines).
    For example, if the data involves dates - the chart type is likely best as a line chart with date on the x-axis.
    Columns: {df.columns.tolist()}. Sample: {df.head(3).to_dict()}
    Your response should be a JSON object in the following format only.
    Structure:
    {{
        "chart_type": "",     // Most appropriate chart type
        "x_axis": "",        // X-axis column name
        "y_axis": "",        // Y-axis column name (if applicable)
        "color": "",         // Color grouping column (optional)
        "title": ""         // Chart title suggestion
    }}
    The chart_type and x_axis are the only required variables.
    """
    return json.loads(cortex.Complete(llm, prompt))

@st.fragment
def plot_dynamic_chart(suggestions, df, message_index):
    defaults = {
                "chart_type": suggestions.get("chart_type", "bar"),
                "x_axis": suggestions.get("x_axis", df.columns[0]),
                "y_axis": suggestions.get("y_axis", df.columns[1] if len(df.columns) > 1 else ""),
                "color": suggestions.get("color", ""),
                "title": suggestions.get("title", "My Chart")
                                }

    col1, col2 = st.columns(2)
    with col1:
        chart_type = st.selectbox(
                "Chart Type",
                ["bar", "line", "scatter", "pie", "histogram", "box", "area"],
                index=["bar", "line", "scatter", "pie", "histogram", "box", "area"].index(
                    defaults["chart_type"] if defaults["chart_type"] in 
                    ["bar", "line", "scatter", "pie", "histogram", "box", "area"] 
                    else "bar"
                ),
            key=f"chert_{message_index}"
            )
            
        x_axis = st.selectbox(
                "X-Axis",
                df.columns,
                index=df.columns.tolist().index(defaults["x_axis"]) 
                if defaults["x_axis"] in df.columns else 0,
                key=f"x_{message_index}"
            )

    with col2:
        y_axis = st.selectbox(
                "Y-Axis" if chart_type not in ["histogram", "pie"] else "Value",
                [None] + df.columns.tolist(),
                index=([None] + df.columns.tolist()).index(defaults["y_axis"])
                if defaults["y_axis"] in df.columns else 0,
            key=f"y_{message_index}" 
            ) if chart_type != "pie" else None

        color = st.selectbox(
                "Color Encoding",
                [None] + df.columns.tolist(),
                index=([None] + df.columns.tolist()).index(defaults["color"]) 
                if defaults["color"] in df.columns else 0,
            key=f"colour_{message_index}"
            )

    title = st.text_input("Chart Title", defaults["title"],
            key=f"title_{message_index}")

    # Generate chart
    chart_map = {
        "bar": px.bar,
        "line": px.line,
        "scatter": px.scatter,
        "pie": px.pie,
        "histogram": px.histogram,
        "box": px.box,
        "area": px.area
    }

    args = {
        "data_frame": df,
        "x": x_axis,
        "y": y_axis if chart_type != "pie" else df.columns[1],
        "color": color,
        "title": title
    }

    if chart_type == "pie":
        args.update({"names": x_axis, "values": y_axis or df.columns[1]})

    fig = chart_map[chart_type](**args)
        
    return st.plotly_chart(fig, use_container_width=True)

    

def show_conversation_history() -> None:
    for message_index, message in enumerate(st.session_state.messages):
        chat_role = "assistant" if message["role"] == "analyst" else "user"
        with st.chat_message(chat_role):
            display_content(
                content=message["content"],
                request_id=message.get("request_id"),
                message_index=message_index,
            )


def display_content(
    content: List[Dict[str, str]],
    request_id: Optional[str] = None,
    message_index: Optional[int] = None,
) -> None:
    """Displays a content item for a message."""
    message_index = message_index or len(st.session_state.messages)
    #if request_id:
    #    with st.expander("Request ID", expanded=False):
    #        st.markdown(request_id)
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
            #st.markdown(cortex.Complete('llama3.1-8b',f'rewrite this in the style of Donald Trump {item["text"]}'))
        elif item["type"] == "suggestions":
            with st.expander("Suggestions", expanded=True):
                for suggestion_index, suggestion in enumerate(item["suggestions"]):
                    if st.button(suggestion, key=f"{message_index}_{suggestion_index}"):
                        st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            with st.expander("Results", expanded=True):
                with st.spinner("Running SQL..."):
                    
                    df = session.sql(item["statement"]).to_pandas()

                    data_tab, viz_tab, sql_tab = st.tabs(
                                ["🔍 Data", ":sparkles: Visualise", "✏️ SQL"])
                    with data_tab:
                        if df.shape[1] == 1 and df.shape[0] == 1:
                            column_name = df.columns[0]
                            value = df.iloc[0, 0]
                            # Check if the value is not a string
                            if not isinstance(value, str):
                                st.metric(column_name, millify(value, 2))
                            else:
                                # If it's a string, display as a dataframe
                                data_tab.dataframe(df, hide_index=True)

                        else:
                            data_tab.dataframe(df,hide_index=True)
                        
                    with sql_tab:
                        st.code(item["statement"], language="sql")
                        
                with st.spinner("Generating chart..."):            
                    with viz_tab:
                        
                            suggestions = get_chart_suggestions(df)
                            if suggestions:
                                st.session_state.suggestions = suggestions
                                #st.success("Suggestions generated!")
                            else:
                                st.error("Failed to generate suggestions")
                            if "suggestions" in st.session_state:
                                suggestions = st.session_state.suggestions
    
                            plot_dynamic_chart(suggestions, df, message_index)
                              

###############
#  MAIN APP   #
###############
init_session_state()
yaml_interactive_selector()  

with stylable_container(key='main_container', css_styles="""{border-radius: 15px; gap: 10px; padding: 30px; background-color: rgba(255,255,255,0.8);} button{background-color: White;border-color: #2A2725}; table, th, td{border-color: #2A2725; border: 0.2px solid}"""):
    
    c1, c2 = st.columns([1000,1]) # Use small dummy second column to avoid objects pushing off right edge of container
    with c1:      
        st.image(vLogo, width=int(100))
        st.title("Cortex Analyst")
        st.write("💬   Welcome to Snowflake's text-to-sql solution. Simply select the data domain in the sidebar and begin asking questions. Nothing is more ubiquitous than chat.") 
        st.divider()
        
        show_conversation_history()

        if st.session_state.active_suggestion:
            process_message(prompt=st.session_state.active_suggestion)
            st.session_state.active_suggestion = None


with stylable_container(key='Enter chat',css_styles="""{border-radius: 15px; gap: 10px; padding: 20px;}"""):
    if user_input := st.chat_input("What is your question?"):
        with c1: 
            process_message(prompt=user_input)
    if st.button("Reset conversation"):
        st.session_state.clear()
        init_session_state()
        
        
