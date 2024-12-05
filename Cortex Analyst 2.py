import streamlit as st
import _snowflake
import snowflake.cortex as cortex
import json
import math
from streamlit_extras.stylable_container import stylable_container
from snowflake.snowpark.context import get_active_session
import plotly.express as px
from typing import Dict, List, Optional, Callable, Literal, Set, Tuple, TypedDict
from dataclasses import dataclass, field
from snowflake.snowpark.exceptions import SnowparkSQLException
import pandas as pd
from plotly.graph_objs._figure import Figure

session = get_active_session()

###############
# APP SETTING #
###############

# SELECT WHICH MODEL YOU'D LIKE TO USE TO GENERATE CHARTS
llm_assistant = "llama3.1-8b"

# FILL THE SECTION IN BELOW WITH DETAILS ON AVAILABLE SEMANTIC MODELS 
def yaml_interactive_selector():    
    models = [
        {"name": "**Movie Model**", "icon": "🎬", "description": "Wondering what to watch tonight? Why not check out the IMDB dataset.","file":"LLM_DEMO.ANALYST.SEMANTIC/movie_model_lsq.yaml","table":"LLM_DEMO.ANALYST.MOVIE_FULL"},
        {"name": "**Sales Model**", "icon": "💰", "description": "Your classic business example - ask data of fictional sales data.","file":"CORTEX_ANALYST_DEMO.REVENUE_TIMESERIES.RAW_DATA/revenue_timeseries.yaml","table":"CORTEX_ANALYST_DEMO.REVENUE_TIMESERIES.DAILY_REVENUE_BY_PRODUCT"},
        {"name": "**Whisky Model**", "icon": "🥃", "description": "Quench your thirst with the Meta-Critic Whisky Database.","file":"LLM_DEMO.ANALYST.SEMANTIC/whisky.yaml","table":"LLM_DEMO.ANALYST.WHISKY_TABLE"}
    ]

    # Initialize session state for selected model if not already set
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    
    for model in models:
        with st.sidebar.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                #st.write(model['icon'])
                with st.popover(model['icon']):
                    q = f"SELECT * FROM {model['table']} LIMIT 5"
                    data_df = session.sql(q)
                    st.dataframe(data_df)
            with col2:
                st.write(model['name'])
                st.caption(model['description'])
            
            if st.sidebar.button(f"Select {model['name']}", key=f"model_{model['name']}"):
                st.session_state.selected_model = model['file']
                st.sidebar.success(f"Selected: {model['name']}")

    return st.session_state.selected_model

###############
# STYLE SHEET #
###############

#HERE ARE THE VISUAL ELEMENTS, IGNORE THIS UNLESS YOU'D LIKE TO ALTER THE STYLE
st.write('<style>div.block-container{margin-top:0;padding-top:0;position:relative;top:-50px}</style>', unsafe_allow_html=True)
vLogo = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Snowflake_Logo.svg/1024px-Snowflake_Logo.svg.png'      

#Set the background colour here
st.markdown("<style>[data-testid=stAppViewContainer] {background-color: #F2F0E9}</style>", unsafe_allow_html=True)

# Set all text colors (including for st.header, st.subheader, st.write, st.button, etc)
st.markdown("<style> h1, h2, h3, h4, h5, h6, a, p, span, ul, li, ol{color: #2A2725;} </style>", unsafe_allow_html=True)

# Set sidebar background color
st.markdown("<style>[data-testid=stSidebar] {background-color: #E6E3D8}</style>", unsafe_allow_html=True)


###############
#  CORE CODE  #
###############

st.sidebar.title("Semantic Model Selector")    

selected_model = yaml_interactive_selector()
    
if selected_model:
        # Use the selected YAML file in your main application logic
    st.info(f"**Current Model**: {selected_model}")
        # Update your semantic model loading logic
    semantic_model_file = f"@{selected_model}"
else:
    st.warning('No data domain selected, pick one in the sidebar!', icon="⚠️")

def send_message() -> dict:
    """Calls the REST API and returns the response."""
    
    request_body = {
        "messages": st.session_state.messages,
        "semantic_model_file": f"{semantic_model_file}",
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


def message_idx_to_question(idx: int) -> str:
    """
    Retrieve the question text from a message in the session state based on its index.

    This function checks the role of the message and returns the appropriate text:
    * If the message is from the user, it returns the prompt content
    * If the message is from the analyst and contains an interpretation of the question, it returns
    the interpreted question
    * Otherwise, it returns the previous user prompt

    Args:
        idx (int): The index of the message in the session state.

    Returns:
        str: The question text extracted from the message.
    """
    # Add a check to ensure the index is valid
    if idx < 0 or idx >= len(st.session_state.messages):
        # If the index is out of range, return a default or the last message's text
        if st.session_state.messages:
            return str(st.session_state.messages[-1]["content"][0]["text"])
        return "No previous message found"

    msg = st.session_state.messages[idx]

    # if it's user message, just return prompt content
    if msg["role"] == "user":
        return str(msg["content"][0]["text"])

    # If it's analyst response, if it's possibleget question interpretation from Analyst
    if msg["content"][0]["text"].startswith(
        "This is our interpretation of your question:"
    ):
        return str(
            msg["content"][0]["text"]
            .strip("This is our interpretation of your question:\n")
            .strip("\n")
            .strip("_")
        )

    # Else just return previous user prompt
    return str(st.session_state.messages[idx - 1]["content"][0]["text"])

def millify(n, decimal_places=0):
    millnames = ['',' K',' M',' B',' T']
    n = float(n)
    millidx = max(0, min(len(millnames)-1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    format_string = '{:.' + str(decimal_places) + 'f}{}'
    return format_string.format(n / 10**(3 * millidx), millnames[millidx])

@dataclass
class ChartDefinition:
    """
    A data class to store configuration details for a chart.

    This class is designed to encapsulate the configuration details required to generate
    charts using plotly.express functions. It stores parameter names and other relevant
    information needed to create various types of charts.

    Attributes:
        name (str): The name of the chart.
        plotly_fnc (Callable): The Plotly function to generate the chart.
        icon (str): An icon representing the chart type.
        base_column_args (List[str]): Names of the base Plotly arguments that take column names as input.
        extra_column_args (List[str]): Additional Plotly arguments that take column names as input.
        additional_params (List[str]): Other plot-specific parameter names.
    """

    name: str
    plotly_fnc: Callable
    icon: str
    base_column_args: List[str] = field(default_factory=lambda: ["x", "y"])
    extra_column_args: List[str] = field(default_factory=list)
    additional_params: List[str] = field(default_factory=list)

    def get_pretty_name(self) -> str:
        """Get name with icon."""
        return f"{self.name} {self.icon}"


class ChartParams(TypedDict, total=False):
    """
    A dict containing all supported parameters for configuring charts using Plotly Express.

    This dictionary is used to store the parameters required by various Plotly Express chart plotting functions.
    Each key represents a parameter name, and the corresponding value represents the parameter value.
    """

    data_frame: Optional[pd.DataFrame]

    x: Optional[str]
    y: Optional[str]
    names: Optional[str]
    values: Optional[str]
    color: Optional[str]

    barmode: Optional[str]
    orientation: Optional[str]
    nbins: Optional[int]
    line_shape: Optional[str]


class ChartConfigDict(TypedDict):
    """A dict containing all configuration required to draw a chart."""

    type: str
    params: ChartParams


ALL_SUPPORTED_ARGS: Dict[str, Literal["column", "number", "option"]] = {
    "x": "column",
    "y": "column",
    "names": "column",
    "values": "column",
    "color": "column",
    "barmode": "option",
    "orientation": "option",
    "line_shape": "option",
    "nbins": "number",
}

ALL_SUPPORTED_OPTIONS: Dict[str, List[str]] = {
    "barmode": ["relative", "group", "stack", "overlay"],
    "orientation": ["v", "h"],
    "line_shape": ["linear", "spline", "hv", "vh", "hvh", "vhv"],
}

# A dictionary of all currently supported charts
AVAILABLE_CHARTS: Dict[str, ChartDefinition] = {
    "Bar": ChartDefinition(
        name="Bar Chart",
        plotly_fnc=px.bar,
        icon="📊",
        extra_column_args=["color"],
        additional_params=["barmode", "orientation"],
    ),
    "Line": ChartDefinition(
        name="Line Chart",
        plotly_fnc=px.line,
        icon="📈",
        extra_column_args=["color"],
        additional_params=["line_shape"],
    ),
    "Pie": ChartDefinition(
        name="Pie Chart",
        plotly_fnc=px.pie,
        icon="🥧",
        base_column_args=["names", "values"],
        extra_column_args=["color"],
    ),
    "Histogram": ChartDefinition(
        name="Histogram",
        plotly_fnc=px.histogram,
        icon="📊",
        extra_column_args=["color"],
        additional_params=["nbins"],
    ),
}


def get_all_supported_plotly_args() -> Dict[str, List[str]]:
    """Get all supported plotly args based on all supported charts definitions."""
    base_columns_args = {
        arg for c in AVAILABLE_CHARTS.values() for arg in c.base_column_args
    }
    extra_columns_args = {
        arg for c in AVAILABLE_CHARTS.values() for arg in c.extra_column_args
    }
    additional_params_args = {
        arg for c in AVAILABLE_CHARTS.values() for arg in c.additional_params
    }
    return {
        "base_columns": list(base_columns_args),
        "extra_columns": list(extra_columns_args),
        "additional_params": list(additional_params_args),
    }


def plotly_fig_from_config(df: pd.DataFrame, cfg: ChartConfigDict) -> Figure:
    """
    Generate a Plotly figure based on the provided configuration.

    This function takes a DataFrame and a configuration dictionary, extracts the chart type and parameters,
    and uses the corresponding Plotly function to generate the chart.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be plotted.
        cfg (Dict): A dictionary containing the chart configuration, including the chart type and parameters.

    Returns:
       Figure: The generated Plotly figure.
    """
    chart_name = cfg["type"]
    plt_args = cfg["params"].copy()
    plt_args["data_frame"] = df
    chart_cfg = AVAILABLE_CHARTS[chart_name]
    return chart_cfg.plotly_fnc(**plt_args)


def try_to_parse_raw_response_to_chart_cfg(
    raw_resp: Dict, valid_col_names: Set[str]
) -> Tuple[Optional[ChartConfigDict], Optional[str]]:
    """
    Try to parse a dictionary to a chart configuration object.

    This function validates the input dictionary against predefined rules and
    converts it into a ChartConfigDict object if all validations pass.

    Args:
        raw_resp (Dict): The input dictionary containing chart configuration data.
        valid_col_names (Set[str]): A set of valid column names for validation.

    Returns:
        Tuple[Optional[ChartConfigDict], Optional[str]]: A tuple containing the
        ChartConfigDict object if the input is valid, and None otherwise. The
        second element of the tuple is an error message if validation fails,
        or None if validation passes.
    """
    if not isinstance(raw_resp, dict):
        return None, f"expected a dict but got {type(raw_resp)}"
    chart_type = raw_resp.get("type")
    if chart_type is None:
        return None, "missing required 'type' key"
    if chart_type not in AVAILABLE_CHARTS:
        return None, f"Got chart type '{chart_type}' which doesn't seem to be supported"
    params_dict = raw_resp.copy()
    params_dict.pop("type")
    for param_name, param_value in params_dict.items():
        if param_name not in ALL_SUPPORTED_ARGS:
            return None, f"Param '{param_name}' doesn't seem to be supported"
        param_type = ALL_SUPPORTED_ARGS[param_name]
        if param_type == "column":
            if not isinstance(param_value, str):
                return (
                    None,
                    f"Column param '{param_name}' is expected to be of type str, but found '{type(param_value)}''",
                )
            if param_value not in valid_col_names:
                return (
                    None,
                    f"Column param '{param_name}' does not contain valud column name: '{param_value}'",
                )
        elif param_type == "option":
            allowed_values = ALL_SUPPORTED_OPTIONS[param_name]
            if param_value not in allowed_values:
                return (
                    None,
                    f"Param '{param_name}' contain invalid value: '{param_value}'. Allowed values: {allowed_values}",
                )
        elif param_type == "number":
            if not str(param_value).isnumeric():
                return (
                    None,
                    f"Numeric param '{param_name}' is expected to hold numeric value",
                )
    return ChartConfigDict(type=chart_type, params=ChartParams(**params_dict)), None


def _format_barmode(x: str) -> str:
    return {
        "relative": "Relative",
        "group": "Group",
        "stack": "Stack",
        "overlay": "Overlay",
    }.get(x, x)


def _pick_barmode(component_idx: int = -1, default: str = "relative") -> str:
    options = ALL_SUPPORTED_OPTIONS["barmode"]
    default_idx = options.index(default)
    return st.selectbox(
        "Barmode",
        options=options,
        index=default_idx,
        format_func=_format_barmode,
        help="Choose how bars are displayed",
        key=f"barmode_picker_{component_idx}",
    )


def _format_orientation(x: str) -> Optional[str]:
    return "Vertical" if x == "v" else "Horizontal"


def _pick_orientation(component_idx: int = -1, default: str = "v") -> str:
    options = ALL_SUPPORTED_OPTIONS["orientation"]
    default_idx = options.index(default)
    return st.selectbox(
        "Orientation",
        options=options,
        index=default_idx,
        format_func=_format_orientation,
        help="Choose the orientation of the plot",
        key=f"orientation_picker_{component_idx}",
    )


def _pick_nbins(component_idx: int = -1, default: int = 10) -> int:
    return st.slider(
        "Number of bins",
        min_value=1,
        max_value=100,
        value=default,
        help="Choose the number of bins for the histogram",
        key=f"nbins_picker_{component_idx}",
    )


def _format_line_shape(x: str) -> str:
    return {
        "linear": "Linear",
        "spline": "Spline",
        "hv": "Horizontal-Vertical",
        "vh": "Vertical-Horizontal",
        "hvh": "Horizontal-Vertical-Horizontal",
        "vhv": "Vertical-Horizontal-Vertical",
    }.get(x, x)


def _pick_line_shape(component_idx: int = -1, default: str = "linear") -> Optional[str]:
    options = ALL_SUPPORTED_OPTIONS["line_shape"]
    default_idx = options.index(default)
    return st.selectbox(
        "Line shape",
        options=options,
        index=default_idx,
        format_func=_format_line_shape,
        help="Choose the shape of the lines in the plot",
        key=f"line_shape_picker_{component_idx}",
    )


# map extra param name to function rendering UI element
extra_params_pickers: Dict[str, Callable[[int, str], str]] = {
    "barmode": _pick_barmode,
    "orientation": _pick_orientation,
    "nbins": _pick_nbins,
    "line_shape": _pick_line_shape,
}


def _get_default_idx(option: Optional[str], all_options: List[str], idx: int) -> int:
    if option is None:
        return idx
    try:
        return all_options.index(option)
    except ValueError:
        return idx


def chart_picker(
    df: pd.DataFrame,
    default_config: Optional[ChartConfigDict] = None,
    component_idx: int = -1,
) -> Dict:
    """
    Based on provided dataframe and selected chart type will render chart picking controls, and return chart config.

    Args:
        df (pd.DataFrame): The query results.
        default_config Optional[ChartConfigDict] : default config describing how the default values picks for all controls
        component_idx (int): Unique component for index - handy for spawning widgets with unique keys when spawning this component multiple times.

    Returns:
        (Dict): Dict config defining chart
    """
    if not default_config:
        default_config = {"type": "Bar", "params": {}}
    all_col_names = list(df.columns)
    all_charts_names = list(AVAILABLE_CHARTS.keys())
    df_has_at_least_two_columns = len(df.columns) >= 2

    plot_cfg: Dict = {"type": None, "params": {}}
    def_params = default_config["params"]

    # Get default chart type index
    default_chart_name = default_config.get("type")
    if default_chart_name is None:
        default_chart_name = all_charts_names[0]
    default_chart_idx = all_charts_names.index(default_chart_name)

    # Pick chart type
    picked_chart = st.selectbox(
        "Select chart type",
        options=all_charts_names,
        format_func=lambda chart_name: AVAILABLE_CHARTS[chart_name].get_pretty_name(),
        key=f"chart_type_{component_idx}",
        index=default_chart_idx,
        disabled=(not df_has_at_least_two_columns),
    )
    if not df_has_at_least_two_columns:
        st.info("At least two columns are required to plot a chart")
        return plot_cfg

    plot_cfg["type"] = picked_chart
    picked_params = {}
    picked_chart_setings: ChartDefinition = AVAILABLE_CHARTS[picked_chart]

    # We always need 2 columns for x and y
    col1, col2 = st.columns(2)
    col1_name, col2_name = picked_chart_setings.base_column_args

    # Get default indexes
    col1_default_idx = _get_default_idx(def_params.get(col1_name), all_col_names, 0)
    col2_default_idx = _get_default_idx(def_params.get(col2_name), all_col_names, 1)

    picked_params[col1_name] = col1.selectbox(
        col1_name,
        all_col_names,
        key=f"x_col_select_{component_idx}",
        index=col1_default_idx,
    )

    picked_params[col2_name] = col2.selectbox(
        col2_name,
        all_col_names,
        key=f"y_col_select_{component_idx}",
        index=col2_default_idx,
    )

    # Now we can pick more columns
    if len(df.columns) > 2:
        for col_param_name in picked_chart_setings.extra_column_args:
            def_idx = _get_default_idx(
                def_params.get(col_param_name), all_col_names, None
            )
            selectbox_args = {
                "label": col_param_name,
                "options": all_col_names,
                "key": f"{col_param_name}_select_{component_idx}",
            }
            if def_idx is not None:
                selectbox_args["index"] = def_idx
            picked_params[col_param_name] = st.selectbox(**selectbox_args)

    # Other, non-column args
    for param_name in picked_chart_setings.additional_params:
        picker_fnc = extra_params_pickers.get(param_name)
        if picker_fnc is None:
            continue
        fnc_args = {"component_idx": component_idx}
        default_val = default_config["params"].get(param_name)
        if default_val is not None:
            fnc_args["default"] = default_val
        picked_params[param_name] = picker_fnc(**fnc_args)

    plot_cfg["params"] = picked_params
    return plot_cfg

@st.experimental_fragment
def show_chart_tab(df: pd.DataFrame, message_index: int):
    default_plot_cfg = None
    suggested_charts_memory = st.session_state.get("suggested_charts_memory")
    if suggested_charts_memory:
        default_plot_cfg = suggested_charts_memory.get(message_index)

    else:
        st.session_state["suggested_charts_memory"] = {}

    plot_cfg = chart_picker(
        df, default_config=default_plot_cfg, component_idx=message_index
    )
    if plot_cfg.get("type") is not None:
        plotly_fig = plotly_fig_from_config(df, plot_cfg).update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        # On the first run, try to get config suggestion
        if message_index not in st.session_state["suggested_charts_memory"]:
            with st.spinner("Generating chart suggestion..."):
                question = message_idx_to_question(message_index)
                chart_suggestion, _ = get_chart_suggestion(question, df)
                st.session_state["suggested_charts_memory"][
                    message_index
                ] = chart_suggestion
        plotly_fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=-0.20
        ))
        st.plotly_chart(plotly_fig)

    return plot_cfg

def get_chart_suggestion(
    question: str, df: pd.DataFrame
) -> Tuple[Optional[ChartConfigDict], Optional[str]]:
    """Generate the best suitable config for given question and execution result."""
    df_as_text = df.head(10).to_string(index=False)
    session = get_active_session()
    prompt = f"""
Your task is to provide suggestion for the best possible chart answering the following question: {question}. 
If the user mentions a chart type in their ask, bias towards that chart type, unless it's unavailable.
Here are results after running query which answers this question:

<RESULTS_BEGIN>
{df_as_text}
<RESULTS_END>

Your suggestion will take the form of JSON object representing plot config, in this format:
{{
    "type": <chart-type>,
    "<param-1>": "<param-1 value>",
    "<param-2>": "<param-2 value>",
    ...
}}
All params, beside first "type" key, represent plotly.express function arguments.
Here is the list of all types and params and plot functions supported by the whole system:
{_get_all_supported_params_instruction}

So if you would like to suggest pie-chart, with:
* "REGION" column as names
* "TOTAL_REVENUE" as values
you should output this:
{{
    "type": "Pie",
    "names": "REGION",
    "values": "TOTAL_REVENUE"
}}
Restrict yourself only to use those params:
{', '.join(ALL_SUPPORTED_ARGS.keys())}
Be aware that 'nbins' param must take numeric value, 'auto' is not allowed there.

Now generate your answer, do not add any additional text, only JSON object:
"""
    try:
        chart_suggestion_response_raw = cortex.Complete(
            model=llm_assistant, prompt=prompt, session=session
        )
        parsed_response = json.loads(chart_suggestion_response_raw)
    except SnowparkSQLException as e:
        err_msg = f"Error while generating suggestions through cortex.Complete: {e}"
        return None, err_msg
    except json.JSONDecodeError as e:
        err_msg = f"Error while parsing reponse from cortex.Compleate: {e}"
        return None, err_msg

    out_cfg, error = try_to_parse_raw_response_to_chart_cfg(
        parsed_response, set(df.columns)
    )
    if out_cfg is None:
        err_msg = f"Generated plot config is invalid: {error}"
        return None, err_msg

    return out_cfg, None

def _get_all_supported_params_instruction() -> str:
    all_instructions = []
    for chart_type, chart_def in AVAILABLE_CHARTS.values:
        chart: ChartDefinition = chart_def
        available_chart_options = f"type: {chart_type}"
        required_colum_params = (
            f"required column params: {', '.join(chart.base_column_args)}"
        )
        extra_column_params = (
            f"extra column parameters: {', '.join(chart.extra_column_args)}"
        )
        other_params = (
            f"other, non-column parameters: {', '.join(chart.extra_column_args)}"
        )
        all_instructions.append(
            f"""
{available_chart_options}
{required_colum_params}
{extra_column_params}
{other_params}
"""
        )
    return "\n\n".join(all_instructions)


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
                            
                    with viz_tab:
                        plot_cfg = show_chart_tab(df, message_index)

                            


# Set container style
with stylable_container(key='myContainer',css_styles="""
    {border-radius: 15px; gap: 10px; padding: 30px; 
    background-color: rgba(255,255,255,0.8); 
    button{background-color: White;border-color: #2A2725};
    table, th, td{color: #2A2725; border-color: #2A2725; border: 0.2px solid; font-size: 14px;};
    }
    """):
    # Use small dummy second column to avoid objects pushing off right edge of container
    c1, c2 = st.columns([1000,1])
    with c1:
        
        
        ################################################
        ##         Main application code below        ##
        ################################################
        c3,c4,c5 = st.columns([1,4,2])
        c3.image(vLogo, width=int(100))
        c5.write("**Welcome, " + st.experimental_user.user_name.upper() + "!**")
        st.title("Cortex Analyst")
        st.write("💬   Welcome to Snowflake's text-to-sql solution. Simply select the data domain in the sidebar and begin asking questions. Nothing is more ubiquitous than chat.")

        
        st.divider()
        
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.suggestions = []
            st.session_state.active_suggestion = None
        
        
        show_conversation_history()

        if st.session_state.active_suggestion:
            process_message(prompt=st.session_state.active_suggestion)
            st.session_state.active_suggestion = None


with stylable_container(key='Enter chat',css_styles="""{border-radius: 15px; gap: 10px; padding: 20px;}"""):
    if user_input := st.chat_input("What is your question?"):
        with c1: 
            process_message(prompt=user_input)

    c6,c7,c8 = st.columns([1,1,1])
    if c8.button("Reset conversation") or "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.suggestions = []
        st.session_state.active_suggestion = None
        st.session_state.selected_model = None
        
        