import plotly
import os, sys
import base64
import sklearn
import numpy as np
import pandas as pd
import streamlit as st

from loocv import loocv
from util.pages.home_page import *
from s5_fold import *
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go


# Widget for recording
def make_recording_widget(f, widget_values):
    """
    Return a function that wraps a streamlit widget and records the
    widget's values to a global dictionary.
    """

    def wrapper(label, *args, **kwargs):
        widget_value = f(label, *args, **kwargs)
        widget_values[label] = widget_value
        return widget_value

    return wrapper


# Object for dict
class objdict(dict):
    """
    Objdict class to conveniently store a state
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


# Main components
def main_components():
    """
    Expose external CSS and create & return widgets
    """
    # External CSS
    main_external_css = """
        <style>
            hr {margin: 15px 0px !important; background: #ff3a50}
            .footer {position: absolute; height: 50px; bottom: -150px; width:100%; padding:10px; text-align:center; }
            #MainMenu, .reportview-container .main footer {display: none;}
            .btn-outline-secondary {background: #FFF !important}
            .download_link {color: #f63366 !important; text-decoration: none !important; z-index: 99999 !important;
                            cursor:pointer !important; margin: 15px 0px; border: 1px solid #f63366;
                            text-align:center; padding: 8px !important; width: 200px;}
            .download_link:hover {background: #f63366 !important; color: #FFF !important;}
            h1, h2, h3, h4, h5, h6, a, a:visited, .css-9s5bis {color: #f84f57 !important}
            label, stText, p, .caption {color: #035672}
            .css-1adrfps {background: #035672 !important;}
            .streamlit-expanderHeader {font-size: 16px !important;}
            label, stText, .caption, .css-1b0udgb, .css-1inwz65 {color: #FFF !important}
            .tickBarMin, .tickBarMax {color: #f84f57 !important}
            .markdown-text-container p {color: #035672 !important}

            /* Tabs */
            .tabs { position: relative; min-height: 200px; clear: both; margin: 40px auto 0px auto; background: #efefef; box-shadow: 0 48px 80px -32px rgba(0,0,0,0.3); }
            .tab {float: left;}
            .tab label { background: #f84f57; cursor: pointer; font-weight: bold; font-size: 18px; padding: 10px; color: #fff; transition: background 0.1s, color 0.1s; margin-left: -1px; position: relative; left: 1px; top: -29px; z-index: 2; }
            .tab label:hover {background: #035672;}
            .tab [type=radio] { display: none; }
            .content { position: absolute; top: -1px; left: 0; background: #fff; right: 0; bottom: 0; padding: 30px 20px; transition: opacity .1s linear; opacity: 0; }
            [type=radio]:checked ~ label { background: #035672; color: #fff;}
            [type=radio]:checked ~ label ~ .content { z-index: 1; opacity: 1; }

            /* Feature Importance Plotly Link Color */
            .js-plotly-plot .plotly svg a {color: #f84f57 !important}
        </style>
    """
    st.markdown(main_external_css, unsafe_allow_html=True)

    # Fundemental elements
    widget_values = objdict()
    record_widgets = objdict()

    # Sidebar widgets
    sidebar_elements = {
        "button_": st.sidebar.button,
        "slider_": st.sidebar.slider,
        "number_input_": st.sidebar.number_input,
        "selectbox_": st.sidebar.selectbox,
        "multiselect": st.multiselect,
    }
    for sidebar_key, sidebar_value in sidebar_elements.items():
        record_widgets[sidebar_key] = make_recording_widget(
            sidebar_value, widget_values
        )

    return widget_values, record_widgets


# Generate sidebar elements
def generate_sidebar_elements(state, icon, report, record_widgets):
    slider_ = record_widgets.slider_
    selectbox_ = record_widgets.selectbox_
    number_input_ = record_widgets.number_input_

    # Sidebar -- Image/Title
    st.sidebar.image(
        icon, use_column_width=True, caption="View LPI-KCGCN " + report["omic_learn_version"]
    )

    st.sidebar.markdown(
        "# [Options](https://github.com/OmicEra/OmicLearn/wiki/METHODS)"
    )

    # Sidebar -- Random State
    state["random_seed"] = slider_(
        "Random seed:", min_value=0, max_value=99, value=12
    )

    # Sidebar -- Preprocessing
    st.sidebar.markdown(
        "## [Preprocessing](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing)"
    )
    optimizer = [
         "Adam", "SGD", "Adagrad", "RMSprop", "AdamW",
    ]
    state["optimizer"] = selectbox_("Optimizer:", optimizer)

    # Sidebar -- Feature Selection
    st.sidebar.markdown(
        "## [Hyperparameters Selection](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-2.-Feature-selection)"
    )
    feature_methods = [
        "False",
        "True",
    ]
    state["no-CUDA?"] = selectbox_("no-CUDA?:", feature_methods)

    state["Epochs"] = number_input_(
        "Number of GCN’s iterations(epoch):", value=300, min_value=100, max_value=1000
    )
    state["learning_rate"] = number_input_(
        "Learning rate:", value=0.015, min_value=0.001, max_value=1.0
    )
    state["weight_decay"] = number_input_(
        "Weight_decay:", value=1e-7, min_value=1e-8, max_value=1e-5
    )
    state["hidden"] = number_input_(
        "Hidden layer dimension(hidden):", value=250, min_value=100, max_value=1000
    )
    state["alpha"] = number_input_(
        "Alpha:", value=0.62, min_value=0.1, max_value=1.0
    )
    state["beta"] = number_input_(
        "Beta:", value=1.5, min_value=0.0, max_value=5.0
    )
    # Sidebar -- Cross-Validation
    st.sidebar.markdown(
        "## [Cross-validation](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-5.-Validation#4-1-cross-validation)"
    )
    state["cv_method"] = selectbox_(
        "Specify CV method:",
        ["Repeated-K-Fold", "Leave-One-Out Cross-Validation(LOOCV)"],
    )
    state["cv_splits"] = number_input_("CV Splits:", min_value=2, max_value=10, value=5)

    return state


def generate_sidebar_elements1(state, icon, report, record_widgets):
    st.sidebar.image(
        icon, use_column_width=True, caption="View LPI-KCGCN " + report["omic_learn_version"]
    )

    st.sidebar.markdown(
        "# [Options](https://github.com/OmicEra/OmicLearn/wiki/METHODS)"
    )

    return state


def generate_sidebar_elements2(state, icon, report, record_widgets):
    slider_ = record_widgets.slider_
    number_input_ = record_widgets.number_input_
    selectbox_ = record_widgets.selectbox_
    st.sidebar.image(
        icon, use_column_width=True, caption="View LPI-KCGCN" + report["omic_learn_version"]
    )

    st.sidebar.markdown(
        "# [Options](https://github.com/OmicEra/OmicLearn/wiki/METHODS)"
    )

    # Sidebar -- Random State
    state["random_seed"] = slider_(
        "Random seed:", min_value=0, max_value=99, value=12
    )

    # Sidebar -- Preprocessing
    st.sidebar.markdown(
        "## [Preprocessing](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-1.-Preprocessing)"
    )
    optimizer = [
        "Adam", "SGD", "Adagrad", "RMSprop", "AdamW",
    ]
    state["optimizer"] = selectbox_("Optimizer:", optimizer)

    # Sidebar -- Feature Selection
    st.sidebar.markdown(
        "## [Hyperparameters Selection](https://github.com/OmicEra/OmicLearn/wiki/METHODS-%7C-2.-Feature-selection)"
    )
    feature_methods = [
        "False",
        "True",
    ]
    state["no-CUDA?"] = selectbox_("no-CUDA?:", feature_methods)

    state["Epochs"] = number_input_(
        "Number of GCN’s iterations(epoch):", value=300, min_value=100, max_value=1000
    )
    state["learning_rate"] = number_input_(
        "Learning rate:", value=0.015, min_value=0.001, max_value=1.0
    )
    state["weight_decay"] = number_input_(
        "Weight_decay:", value=1e-7, min_value=1e-8, max_value=1e-5
    )
    state["hidden"] = number_input_(
        "Hidden layer dimension(hidden):", value=250, min_value=100, max_value=1000
    )
    state["alpha"] = number_input_(
        "Alpha:", value=0.62, min_value=0.1, max_value=1.0
    )
    state["beta"] = number_input_(
        "Beta:", value=1.5, min_value=0.0, max_value=5.0
    )
    Select = [
        "lncRNA", "protein",
    ]
    state["ID"] = selectbox_("ID:", Select)
    if state.ID == "lncRNA":
        state["lncRNAid"] = number_input_(
            "lncRNA ID:", value=1, min_value=1, max_value=990
        )
    else:
        state["proteinid"] = number_input_(
            "protein ID:", value=1, min_value=1, max_value=27
        )
    return state


# Create new list and dict for sessions
@st.cache(allow_output_mutation=True)
def get_sessions():
    return [], {}


# Saving session info
def save_sessions(widget_values, user_name):
    session_no, session_dict = get_sessions()
    session_no.append(len(session_no) + 1)
    session_dict[session_no[-1]] = widget_values
    sessions_df = pd.DataFrame(session_dict)
    sessions_df = sessions_df.T
    sessions_df = sessions_df.drop(
        sessions_df[sessions_df["user"] != user_name].index
    ).reset_index(drop=True)
    new_column_names = {
        k: v.replace(":", "").replace("Select", "")
        for k, v in zip(sessions_df.columns, sessions_df.columns)
    }
    sessions_df = sessions_df.rename(columns=new_column_names)
    sessions_df = sessions_df.drop("user", axis=1)

    st.write("## Session History")
    st.dataframe(
        sessions_df.style.format(precision=3)
    )  # Display only 3 decimal points in UI side
    get_download_link(sessions_df, "session_history.csv")


# Load data
@st.cache(persist=True, show_spinner=True)
def load_data(file_buffer, delimiter):
    """
    Load data to pandas dataframe
    """

    warnings = []
    df = pd.DataFrame()
    if file_buffer is not None:
        if delimiter == "Excel File":
            df = pd.read_excel(file_buffer)

            # check if all columns are strings valid_columns = []
            error = False
            valid_columns = []
            for idx, _ in enumerate(df.columns):
                if isinstance(_, str):
                    valid_columns.append(_)
                else:
                    warnings.append(
                        f"Removing column {idx} with value {_} as type is {type(_)} and not string."
                    )
                    error = True
            if error:
                warnings.append(
                    "Errors detected when importing Excel file. Please check that Excel did not convert protein names to dates."
                )
                df = df[valid_columns]

        elif delimiter == "Comma (,)":
            df = pd.read_csv(file_buffer, sep=",")
        elif delimiter == "Semicolon (;)":
            df = pd.read_csv(file_buffer, sep=";")
        elif delimiter == "Tab (\\t) for TSV":
            df = pd.read_csv(file_buffer, sep="\t")
    return df, warnings


# Show main text and data upload section
def main_text_and_data_upload(state, APP_TITLE):
    st.title(APP_TITLE)

    st.info(
        """
    **Note:** It is possible to get artificially high or low performance because of technical and biological artifacts in the data.
    While View LPI-KCGCN has the functionality to perform K-fold cross validation and local Leave-One-Out Cross-Validation (LOOCV).
    """
    )

    with st.expander("Upload or select sample dataset (*Required)", expanded=True):
        st.info(
            """
            - Upload your excel / csv / tsv file here. Maximum size is 200 Mb.
            - Each row corresponds to a sample, each column to a feature.
            - LncRNA or protein kernels denote to Similarity matrix, if you have any questions about the kernel matrix you can read our paper in detail.
            - Additional features should be marked with a leading '_'.
        """
        )
        file_buffer = st.file_uploader(
            "Upload your dataset below", type=["csv", "xlsx", "xls", "tsv"]
        )
        st.markdown(
            """**Note:** By uploading a file, you agree to our
                    [Apache License](https://github.com/OmicEra/OmicLearn/blob/master/LICENSE).
                    Data that is uploaded via the file uploader will not be saved by us;
                    it is only stored temporarily in RAM to perform the calculations."""
        )

        if file_buffer is not None:
            if file_buffer.name.endswith(".xlsx") or file_buffer.name.endswith(".xls"):
                delimiter = "Excel File"
            elif file_buffer.name.endswith(".tsv"):
                delimiter = "Tab (\\t) for TSV"
            else:
                delimiter = st.selectbox(
                    "Determine the delimiter in your dataset",
                    ["Comma (,)", "Semicolon (;)"],
                )

            df, warnings = load_data(file_buffer, delimiter)

            for warning in warnings:
                st.warning(warning)
            state["df"] = df

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("Or select lncRNA kernel file here:")
        state["sample_file"] = st.selectbox(
            "Or select sample file here:", ["None", "ct", "ep", "swl", "all"]
        )
        st.markdown("Or select  protein kernel file here:")
        state["sample_file_protein"] = st.selectbox(
            "Or select sample file here:", ["None", "go", "ps", "swp", "all"]
        )
        # Sample dataset / uploaded file selection
        dataframe_length = len(state.df)
        # max_df_length = 30

        if state.sample_file != "None" and dataframe_length:
            st.warning(
                "**WARNING:** File uploaded but sample file selected. Please switch sample file to `None` to use your file."
            )
            state["df"] = pd.DataFrame()
            state["dfpro"] = pd.DataFrame()
        elif state.sample_file != "None" and state.sample_file_protein != "None":
            if state.sample_file != "all" and state.sample_file_protein != "all":
                state["df"] = pd.read_excel("data/" + state.sample_file + ".xlsx")
                st.markdown("Using the lncRNA following dataset:")
                st.dataframe(state.df)
                state["dfpro"] = pd.read_excel("data/" + state.sample_file_protein + ".xlsx")
                st.markdown("Using the protein following dataset:")
                st.dataframe(state.dfpro)
        else:
            st.warning("**WARNING:** No dataset uploaded or selected.")

    return state

def main_text_and_data_uploadloocv(state, APP_TITLE):
    st.title(APP_TITLE)

    st.info(
        """
    **Note:** It is possible to get artificially high or low performance because of technical and biological artifacts in the data.
    While View LPI-KCGCN has the functionality to perform K-fold cross validation and local Leave-One-Out Cross-Validation (LOOCV).
    """
    )

    return state
# Prepare system report
def get_system_report():
    """
    Returns the package versions
    """
    report = {}
    report["omic_learn_version"] = "v1.0.0"
    report["python_version"] = sys.version[:5]
    report["pandas_version"] = pd.__version__
    report["numpy_version"] = np.version.version
    report["sklearn_version"] = sklearn.__version__
    report["plotly_version"] = plotly.__version__

    return report


# Generate a download link for Plots and CSV
def get_download_link(exported_object, name):
    """
    Generate download link for charts in SVG and PDF formats and for dataframes in CSV format
    """
    os.makedirs("downloads/", exist_ok=True)
    extension = name.split(".")[-1]

    if extension == "svg":
        exported_object.write_image("downloads/" + name, height=700, width=700, scale=1)
        with open("downloads/" + name) as f:
            svg = f.read()
        b64 = base64.b64encode(svg.encode()).decode()
        href = (
                f'<a class="download_link" href="data:image/svg+xml;base64,%s" download="%s" >Download as *.svg</a>'
                % (b64, name)
        )
        st.markdown("")
        st.markdown(href, unsafe_allow_html=True)

    elif extension == "pdf":
        exported_object.write_image("downloads/" + name, height=700, width=700, scale=1)
        with open("downloads/" + name, "rb") as f:
            pdf = f.read()
        b64 = base64.encodebytes(pdf).decode()
        href = (
                f'<a class="download_link" href="data:application/pdf;base64,%s" download="%s" >Download as *.pdf</a>'
                % (b64, name)
        )
        st.markdown("")
        st.markdown(href, unsafe_allow_html=True)

    elif extension == "csv":
        exported_object.to_csv("downloads/" + name, index=False)
        with open("downloads/" + name, "rb") as f:
            csv = f.read()
        b64 = base64.b64encode(csv).decode()
        href = (
                f'<a class="download_link" href="data:file/csv;base64,%s" download="%s" >Download as *.csv</a>'
                % (b64, name)
        )
        st.markdown("")
        st.markdown(href, unsafe_allow_html=True)

    else:
        raise NotImplementedError("This output format function is not implemented")


# Generate summary text
def generate_text(state, report):
    text = ""
    # Packages
    packages_plain_text = """
        View LPI-KCGCN ({omic_learn_version}) was utilized for performing data analysis, model execution, and creation of plots and charts.
        Machine learning was done in Python ({python_version}). Feature tables were imported via the Pandas package ({pandas_version}) and manipulated using the Numpy package ({numpy_version}).
        The machine learning pipeline was employed using the scikit-learn package ({sklearn_version}).
        The Plotly ({plotly_version}) library was used for plotting.
    """
    text += packages_plain_text.format(**report)

    # Normalization
    if state.normalization == "None":
        text += "No normalization on the data was performed. "
    elif state.normalization in ["StandardScaler", "MinMaxScaler", "RobustScaler"]:
        text += f"Data was normalized in each using a {state.normalization} approach. "
    else:
        params = [f"{k} = {v}" for k, v in state.normalization_params.items()]
        text += f"Data was normalized in each using a {state.normalization} ({' '.join(params)}) approach. "

    # Missing value impt.
    if state.missing_value != "None":
        text += "To impute missing values, a {}-imputation strategy is used. ".format(
            state.missing_value
        )
    else:
        text += "The dataset contained no missing values; hence no imputation was performed. "

    # Features
    if state.feature_method == "None":
        text += "No feature selection algorithm was applied. "
    elif state.feature_method == "ExtraTrees":
        text += "Features were selected using a {} (n_trees={}) strategy with the maximum number of {} features. ".format(
            state.feature_method, state.n_trees, state.max_features
        )
    else:
        text += "Features were selected using a {} strategy with the maximum number of {} features. ".format(
            state.feature_method, state.max_features
        )
    text += "During training, normalization and feature selection was individually performed using the data of each split. "

    # Classification
    params = [f"{k} = {v}" for k, v in state.classifier_params.items()]
    text += f"For classification, we used a {state.classifier}-Classifier ({' '.join(params)}). "

    # Cross-Validation
    if state.cv_method == "RepeatedStratifiedKFold":
        cv_plain_text = """
            When using a repeated (n_repeats={}), stratified cross-validation (RepeatedStratifiedKFold, n_splits={}) approach to classify {} vs. {},
            we achieved a receiver operating characteristic (ROC) with an average AUC (area under the curve) of {:.2f} ({:.2f} std)
            and precision-recall (PR) Curve with an average AUC of {:.2f} ({:.2f} std).
        """
        text += cv_plain_text.format(
            state.cv_repeats,
            state.cv_splits,
            "".join(state.class_0),
            "".join(state.class_1),
            state.summary.loc["mean"]["roc_auc"],
            state.summary.loc["std"]["roc_auc"],
            state.summary.loc["mean"]["pr_auc"],
            state.summary.loc["std"]["pr_auc"],
        )
    else:
        cv_plain_text = """
            When using a {} cross-validation approach (n_splits={}) to classify {} vs. {}, we achieved a receiver operating characteristic (ROC)
            with an average AUC (area under the curve) of {:.2f} ({:.2f} std) and Precision-Recall (PR) Curve with an average AUC of {:.2f} ({:.2f} std).
        """
        text += cv_plain_text.format(
            state.cv_method,
            state.cv_splits,
            "".join(state.class_0),
            "".join(state.class_1),
            state.summary.loc["mean"]["roc_auc"],
            state.summary.loc["std"]["roc_auc"],
            state.summary.loc["mean"]["pr_auc"],
            state.summary.loc["std"]["pr_auc"],
        )

    if state.cohort_column is not None:
        text += "When training on one cohort and predicting on another to classify {} vs. {}, we achieved the following AUCs: ".format(
            "".join(state.class_0), "".join(state.class_1)
        )
        for i, cohort_combo in enumerate(state.cohort_combos):
            text += "{:.2f} when training on {} and predicting on {} ".format(
                state.cohort_results["roc_auc"][i], cohort_combo[0], cohort_combo[1]
            )
            text += ", and {:.2f} for PR Curve when training on {} and predicting on {}. ".format(
                state.cohort_results["pr_auc"][i], cohort_combo[0], cohort_combo[1]
            )

    # Print the all text
    st.header("Summary")
    with st.expander("Summary text"):
        st.info(text)


# Generate footer
def generate_footer_parts(report):
    # Citations
    citations = """
        <br> <b>APA Format:</b> <br>
        Cong Shen(Member, IEEE), Dongdong Mao , Jijun Tang , Zhijun Liao ,and Shengyong Chen (Senior Member, IEEE) (2023 IEEE JBHI).
        Prediction of LncRNA-Protein Interactions Based on Kernel Combinations and Graph Convolutional Networks</a>.
    """

    # Put the footer with tabs
    footer_parts_html = """
        <div class="tabs">
            <div class="tab"> <input type="radio" id="tab-1" name="tab-group-1" checked> <label for="tab-1">Citations</label> <div class="content"> <p> {} </p> </div> </div>
            <div class="tab"> <input type="radio" id="tab-2" name="tab-group-1"> <label for="tab-2">Report bugs</label> <div class="content">
                <p><br>
                    We appreciate all contributions. 👍 <br>
                    You can report bugs or request a feature using the link below or sending us an e-mail:
                    <br><br>
                    <a class="download_link" href="https://github.com/OmicEra/OmicLearn/issues/new/choose" target="_blank">Report a bug via GitHub</a>
                    <a class="download_link" href="mailto:info@omicera.com">Report a bug via Email</a>
                </p>
            </div> </div>
        </div>

        <div class="footer">
            <i> View LPI-KCGCN {} </i> <br> <img src="https://omicera.com/wp-content/uploads/2020/05/cropped-oe-favicon-32x32.jpg" alt="OmicEra Diagnostics GmbH">
            <a href="https://omicera.com" target="_blank">OmicEra</a>.
        </div>
        """.format(
        citations, report["omic_learn_version"]
    )

    st.write("## Cite us & Report bugs")
    st.markdown(footer_parts_html, unsafe_allow_html=True)


def restruct(state):
    rnafeat = np.loadtxt('./dataset1/' + state.sample_file + '.csv', delimiter=',')
    protfeat = np.loadtxt('./dataset1/' + state.sample_file_protein + '.csv', delimiter=',')
    gl = norm_adj(rnafeat)
    gp = norm_adj(protfeat)
    return gl, gp


def pictureroc(state):
    if state.sample_file != "all" and state.sample_file_protein != "all":
        rocfile = './figure/roc_' + state.sample_file + '_' + state.sample_file_protein + '.txt'
    else:
        rocfile = './figure/fusion_roc.txt'
    rocdata = np.loadtxt(rocfile, delimiter=',')
    fpr = rocdata[0]
    tpr = rocdata[1]
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={state.auc :.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig


def picturepr(state):
    if state.sample_file != "all" and state.sample_file_protein != "all":
        prfile = './figure/pr_' + state.sample_file + '_' + state.sample_file_protein + '.txt'
    else:
        prfile = './figure/fusion_pr.txt'
    rocdata = np.loadtxt(prfile, delimiter=',')
    recall = rocdata[0]
    precision = rocdata[1]
    fig = px.area(
        x=recall, y=precision,
        title=f'Precision-Recall Curve ({state.aupr :.4f})',
        labels=dict(x='Recall', y='Precision'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    return fig


def heatmaps(state):
    if state.sample_file != "all" and state.sample_file_protein != "all":
        rnafile = './dataset1/' + state.sample_file + '.csv'
        profile = './dataset1/' + state.sample_file_protein + '.csv'

    rnadata = np.loadtxt(rnafile, delimiter=',')
    prodata = np.loadtxt(profile, delimiter=',')
    fig1 = px.imshow(rnadata, text_auto=True)
    fig2 = px.imshow(prodata, text_auto=True)

    return fig1, fig2


def heatmaps2(state, rnadf, prodf):
    global fig1, fig2
    if state.sample_file != "all" and state.sample_file_protein != "all":
        # rnafile = './dataset1/' + state.sample_file + '.csv'
        # profile = './dataset1/' + state.sample_file_protein + '.csv'
        #
        # rnadata = np.loadtxt(rnafile, delimiter=',')
        # prodata = np.loadtxt(profile, delimiter=',')
        fig1 = px.imshow(rnadf, text_auto=True)
        fig2 = px.imshow(prodf, text_auto=True)

    return fig1, fig2


def threeD(state):
    z_data = pd.read_csv('comp.csv', header=None)
    z_data.iloc[:, 5] = np.array([state.auc, state.aupr, state.recall, state.precision, state.f1])
    z = z_data.values
    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    fig = go.Figure(data=[go.Surface(z=z, x=y, y=x)])
    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90)
                      )

    # Different types of customized ticks
    fig.update_layout(scene=dict(
        xaxis=dict(

            ticktext=['CF', 'LPIHN', 'RWR', 'LPBNI', 'LPIIBNRA', 'Ours', 'LPISKF', 'XGBoost'
                                                                                   'LPIGAC', 'CatBoost',
                      'DRPLPI', 'Random forest', 'LPIDF'],
            tickvals=[y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12],
                      ],
        ),
        yaxis=dict(
            ticktext=['AUC', 'AUPR', 'Recall', 'Precision', 'F1-Score'],
            tickvals=[x[0], x[1], x[2], x[3], x[4]],
        ),
        zaxis=dict(
            nticks=4, ticks='outside',
            tick0=0, tickwidth=4), ),
        width=700,
        margin=dict(r=10, l=10, b=10, t=10)
    )
    return fig


def show_case_study(state):
    if state.ID == "protein":
        file = './casestudy/pro' + str(state.proteinid) + '.csv'
        if not os.path.exists(file):
            loocv(state)
        value = np.loadtxt(file, delimiter=',')
        lncRNAname = np.loadtxt('./dataset1/LncRNAName.txt', usecols=(0), dtype=str)
        proteinname = np.loadtxt('./dataset1/ProteinName.txt', usecols=(0), dtype=str)

        dic = {
            "protein": proteinname[state.proteinid - 1],
            "lncRNA": lncRNAname,
            "fre": value[:, state.proteinid - 1]
        }
        df = pd.DataFrame(dic)
        fig = px.parallel_categories(df, dimensions=['protein', 'lncRNA'],
                                     color="fre", color_continuous_scale=px.colors.sequential.Inferno,
                                     labels={'protein': 'protein', 'lncRNA': 'lncRNA'})
    else:
        file = './casestudy/lnc' + str(state.lncRNAid) + '.csv'
        if not os.path.exists(file):
            loocv(state)
        value = np.loadtxt(file, delimiter=',')
        lncRNAname = np.loadtxt('./dataset1/LncRNAName.txt', usecols=(0), dtype=str)
        proteinname = np.loadtxt('./dataset1/ProteinName.txt', usecols=(0), dtype=str)

        dic = {
            "protein": proteinname,
            "lncRNA": lncRNAname[state.lncRNAid - 1],
            "fre": value[state.lncRNAid - 1, :]
        }
        df = pd.DataFrame(dic)
        fig = px.parallel_categories(df, dimensions=['protein', 'lncRNA'],
                                     color="fre", color_continuous_scale=px.colors.sequential.Inferno,
                                     labels={'protein': 'protein', 'lncRNA': 'lncRNA'})
    return fig
