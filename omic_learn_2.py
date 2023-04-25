"""OmicLearn main file."""
import random
import warnings
import pandas as pd
from PIL import Image
import streamlit as st
from datetime import datetime
from Integrated5_fold import *
from s5_fold import *

warnings.simplefilter("ignore")

# UI components and others func.
from utils.ui_helper import (
    main_components,
    get_system_report,
    save_sessions,
    load_data,
    main_text_and_data_upload,
    objdict,
    generate_sidebar_elements,
    generate_sidebar_elements1,
    get_download_link,
    generate_text,
    generate_footer_parts,
    restruct,
    pictureroc,
    picturepr,
    heatmaps,
    heatmaps2, threeD, generate_sidebar_elements2, show_case_study, main_text_and_data_uploadloocv
)

# Set the configs
APP_TITLE = "View LPI-KCGCN — A visualization platform for LPI-KCGCN"
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=Image.open("./utils/omic_learn.ico"),
    layout="centered",
    initial_sidebar_state="auto",
)
icon = Image.open("./utils/6OIM.png")
report = get_system_report()


# Display results and plots
def classify_and_plot(state):
    state.bar = st.progress(0)
    # Cross-Validation
    st.markdown("Performing analysis and Running cross-validation")
    # cv_results, cv_curves = perform_cross_validation(state)

    st.header("Cross-validation results")

    # ROC-AUC
    with st.expander(
            "Receiver operating characteristic Curve and Precision-Recall Curve"
    ):
        st.subheader("Receiver operating characteristic")
        p = pictureroc(state)
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "roc_curve.pdf")
            get_download_link(p, "roc_curve.svg")

        # Precision-Recall Curve
        st.subheader("Precision-Recall Curve")
        st.markdown(
            "Precision-Recall (PR) Curve might be used for imbalanced datasets."
        )
        p = picturepr(state)
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "pr_curve.pdf")
            get_download_link(p, "pr_curve.svg")



    return state


# Main Function
def OmicLearn_Main():
    # Define state
    state = objdict()
    state["df"] = pd.DataFrame()
    state["class_0"] = None
    state["class_1"] = None

    # Main components
    widget_values, record_widgets = main_components()

    # Welcome text and Data uploading
    state = main_text_and_data_upload(state, APP_TITLE)

    # Checkpoint for whether data uploaded/selected
    # state = checkpoint_for_data_upload(state, record_widgets)

    # Sidebar widgets
    state = generate_sidebar_elements(state, icon, report, record_widgets)

    if (
            # (state.df is not None)
            # and (state.class_0 and state.class_1) and
            (st.button("RUN Cross-validation", key="run"))
    ):

        st.info(
            f"""
            **Running info (this is K-fold Cross Validation)**
            - Using the following lncRNA feature kernel: **`{state.sample_file}`** and protein feature kernel :**`{state.sample_file_protein}`**.
            - Using Cross Validation: **`{state.cv_splits}`** fold.
                   - Using parameters: Random seed:`{state.random_seed}`, Number of epochs to train:`{state.Epochs}`, Learning rate:`{state.learning_rate}`, Weight decay (L2 loss on parameters):`{state.weight_decay}`, Dimension of representations:`{state.hidden}`, Weight between lncRNA space and protein space:`{state.alpha}`, Hyperparameter beta:`{state.beta}`.
            - Note that View LPI-KCGCN is intended to be an exploratory tool to assess the performance of algorithms,
                rather than providing a classification model for production.
        """
        )

        # Session and Run info
        widget_values["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " (UTC)"
        if state.sample_file == "all" and state.sample_file_protein == "all":
            state = intgrated5_fold(state)
        else:
            state = s5_main(state)
        state = classify_and_plot(state)
        generate_footer_parts(report)

    # else:
    #     pass


def OmicLearn_Main1():
    # Define state
    state = objdict()
    state["df"] = pd.DataFrame()
    state["class_0"] = None
    state["class_1"] = None

    # Main components
    widget_values, record_widgets = main_components()
    # Welcome text and Data uploading
    state = main_text_and_data_upload(state, APP_TITLE)
    # Sidebar widgets
    state = generate_sidebar_elements1(state, icon, report, record_widgets)

    if st.button("Show Heatmaps", key="run"):
        with st.expander(
                "lncRNAs-lncRNAs similarity and protein-protein similarity "
        ):
            st.subheader(f"lncRNAs-lncRNAs `{state.sample_file}` similarity")
            p1, p2 = heatmaps(state)
            st.plotly_chart(p1, use_container_width=True)
            if p1:
                get_download_link(p1, "roc_curve.pdf")
                get_download_link(p1, "roc_curve.svg")

            st.subheader(f"lncRNAs-lncRNAs *`{state.sample_file_protein}`* similarity")
            st.markdown(
                "Precision-Recall (PR) Curve might be used for imbalanced datasets."
            )

            st.plotly_chart(p2, use_container_width=True)
            if p2:
                get_download_link(p2, "pr_curve.pdf")
                get_download_link(p2, "pr_curve.svg")

    if st.button("Reconstruct Kernels", key="runs"):
        st.markdown(f"LncRNA **`{state.sample_file}`** reconstructed kernel:")
        rnadf, prodf = restruct(state)
        st.dataframe(rnadf)
        st.markdown(f"Protein **`{state.sample_file_protein}`** restructed kernel:")
        st.dataframe(prodf)
        with st.expander(
                "lncRNAs-lncRNAs restructed similarity and  restructed protein-protein similarity "
        ):
            st.subheader(f"lncRNAs-lncRNAs `{state.sample_file}` restructed similarity")
            p3, p4 = heatmaps2(state, rnadf, prodf)
            st.plotly_chart(p3, use_container_width=True)
            if p3:
                get_download_link(p3, "roc_curve.pdf")
                get_download_link(p3, "roc_curve.svg")

            st.subheader(f"lncRNAs-lncRNAs *`{state.sample_file_protein}`* restructed similarity")
            st.markdown(
                "Precision-Recall (PR) Curve might be used for imbalanced datasets."
            )

            st.plotly_chart(p4, use_container_width=True)
            if p4:
                get_download_link(p4, "pr_curve.pdf")
                get_download_link(p4, "pr_curve.svg")


def OmicLearn_Main3():
    # Define state
    state = objdict()
    state["df"] = pd.DataFrame()
    state["class_0"] = None
    state["class_1"] = None

    widget_values, record_widgets = main_components()
    state = main_text_and_data_upload(state, APP_TITLE)
    state = generate_sidebar_elements(state, icon, report, record_widgets)

    if (
            (st.button("RUN", key="run"))
    ):

        st.info(
            f"""
                   **Running info (this is Compare to The-Staet-of-Art)**
                   - The-Staet-of-Art Works:'CF', 'LPIHN', 'RWR', 'LPBNI', 'LPIIBNRA', 'LPISKF', 'XGBoost' 'LPIGAC', 'CatBoost'
                   - Using the following lncRNA feature kernel: **`{state.sample_file}`** and protein feature kernel :**`{state.sample_file_protein}`**.
                   - Using Cross Validation: **`{state.cv_splits}`** fold.
                   - Using parameters: Random seed:`{state.random_seed}`, Number of epochs to train:`{state.Epochs}`, Learning rate:`{state.learning_rate}`, Weight decay (L2 loss on parameters):`{state.weight_decay}`, Dimension of representations:`{state.hidden}`, Weight between lncRNA space and protein space:`{state.alpha}`, Hyperparameter beta:`{state.beta}`.
                   - Note that View LPI-KCGCN is intended to be an exploratory tool to assess the performance of algorithms,
                       rather than providing a classification model for production.
               """
        )
        widget_values["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " (UTC)"
        if state.sample_file == "all" and state.sample_file_protein == "all":
            state = intgrated5_fold(state)
        else:
            state = s5_main(state)
        with st.expander(
                "Compare to the State-of-the-art-work"
        ):
            st.subheader("Receiver operating characteristic")
            p = threeD(state)
            st.plotly_chart(p, use_container_width=True)
            if p:
                get_download_link(p, "roc_curve.pdf")
                get_download_link(p, "roc_curve.svg")

        generate_footer_parts(report)


def OmicLearn_Main4():
    # Define state
    state = objdict()
    state["df"] = pd.DataFrame()
    state["class_0"] = None
    state["class_1"] = None

    widget_values, record_widgets = main_components()
    state = main_text_and_data_uploadloocv(state, APP_TITLE)
    state = generate_sidebar_elements2(state, icon, report, record_widgets)

    if (
            (st.button("RUN loocv", key="run"))
    ):

        st.info(
            f"""
                   **Running info (this is Leave-One-Out Cross-Validation)**
                   - Using Leave-One-Out Cross-Validation(LOOCV).
                   - Using parameters: Random seed:`{state.random_seed}`, Number of epochs to train:`{state.Epochs}`, Learning rate:`{state.learning_rate}`, Weight decay (L2 loss on parameters):`{state.weight_decay}`, Dimension of representations:`{state.hidden}`, Weight between lncRNA space and protein space:`{state.alpha}`, Hyperparameter beta:`{state.beta}`.
                   - Note that View LPI-KCGCN is intended to be an exploratory tool to assess the performance of algorithms,
                       rather than providing a classification model for production.
               """
        )
        if state.ID=="lncRNA":
            st.info(
                f"""
                Select lncRNA ID number:`{state.lncRNAid}`.
                """
            )
        else:
            st.info(
                f"""
                Select Protein ID number:`{state.proteinid}`.
                """
            )
        with st.expander(
                "SEE CASE STUDY"
        ):
            st.subheader("case study")
            p = show_case_study(state)
            st.plotly_chart(p, use_container_width=True)
            if p:
                get_download_link(p, "roc_curve.pdf")
                get_download_link(p, "roc_curve.svg")

        # generate_footer_parts(report)
