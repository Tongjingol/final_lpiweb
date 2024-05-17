"""OmicLearn main file."""
import os
import random
import warnings
import pandas as pd
from PIL import Image
import streamlit as st
from datetime import datetime
from Integrated5_fold import *
from krenel import save_kernel_matrix, generate_kernel_matrix, save_metadata, file_exists
from s5_fold2 import *


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
    heatmaps2, threeD, generate_sidebar_elements2, show_case_study, main_text_and_data_uploadloocv,
    generate_sidebar_elements3, main_text_and_data_uploadkernel
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
    st.write(state)
    if (
            # (state.df is not None)
            # and (state.class_0 and state.class_1) and
            (st.button("RUN Cross-validation", key="run"))
    ):
        # if state.sample_file == "None" and state.sample_file_protein is not None:
        #     state.sample_file = state.file_name
        #
        # elif state.sample_file_protein == "None" and state.sample_file is not None:
        #     state.sample_file_protein = state.file_name

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
    #   pass


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
        rnadf = restruct(state)
        st.dataframe(rnadf)
        st.markdown(f"Protein **`{state.sample_file_protein}`** restructed kernel:")
        prodf = restruct(state)
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
            print(p)
            st.plotly_chart(p, use_container_width=True)
            if p:
                get_download_link(p, "roc_curve.pdf")
                get_download_link(p, "roc_curve.svg")

        # generate_footer_parts(report)
def OmicLearn_Main5():
    # Define state
    if "kernel_matrix" not in st.session_state:
        st.session_state.kernel_matrix = None
    if "kernel_type" not in st.session_state:
        st.session_state.kernel_type = None
    state = objdict()
    state["df"] = pd.DataFrame()
    state["class_0"] = None
    state["class_1"] = None
    state["sequences"] = None
    state["sequences_type"] = None

    widget_values, record_widgets = main_components()
    state = main_text_and_data_uploadkernel(state, APP_TITLE)
    state = generate_sidebar_elements3(state, icon, report, record_widgets)

    if state["sequences"] and state["sequences_type"]:
        if state["sequences_type"] == "lncRNA":
            st.info(
                """
                swl--lncRNA Sequential Similarity Kernel;
                ct--lncRNA Sequence Feature Kernel;
            """
            )
            kernel_options = ["Select here", "swl", "ct"]
        elif state["sequences_type"] == "lncRNA_exp":
            st.info(
                """
                ep--lncRNA Expression Kernel;
            """
            )
            kernel_options = ["ep"]
        elif state["sequences_type"] == "protein":
            st.info(
                """
                swp--Protein Sequential Similarity Kernel;
                ps--Protein Sequence Feature Kernel;
            """
            )
            kernel_options = ["Select here", "swp", "ps"]
        elif state["sequences_type"] == "protein_go":
            st.info(
                """
                go--Protein GO Kernel;
            """
            )
            kernel_options = ["go"]
        kernel_type = st.selectbox("Select kernel type", kernel_options)
        # 新增的重置状态逻辑
        if st.session_state.kernel_type != kernel_type:
            st.session_state.kernel_type = kernel_type
            st.session_state.kernel_matrix = None
        sequences = pd.DataFrame(state["sequences"])
        metadata = pd.DataFrame(state["metadata"])
        with st.expander("Sequences", expanded = False):
            if sequences is not None:
                st.write(sequences)
                filenames = os.path.join("newdata",
                                     st.text_input("Enter filename to save the sequences", state["file_name"] + "_" + "sequences" + "." + state["file_extension"]))
                if st.button("Save Sequences", key="save_sequences"):
                    if file_exists(filenames):
                        st.error(f"File {filenames} already exists. Please choose a different name.")
                    else:
                        save_metadata(state["Sequences"], filenames)
                        st.success(f"Sequences saved to {filenames}.")
            if metadata is not None:
                st.write(metadata)
                filenamem = os.path.join("newdata",
                                         st.text_input("Enter filename to save the metadata",
                                                       state["file_name"] + "_" + "metadata" + ".txt"))
                if st.button("Save Metadata", key="save_metadata"):
                    if file_exists(filenames):
                        st.error(f"File {filenames} already exists. Please choose a different name.")
                    else:
                        save_metadata(state["metadata"], filenames)
                        st.success(f"metadata saved to {filenames}.")
        if (
                (st.button("RUN kernel", key="run"))
        ):
            if kernel_type == "Select here":
               st.warning("Please select kernel type !")
            else:
                kernel_matrix = generate_kernel_matrix(state["sequences"], kernel_type, state["sequences_type"])
                st.session_state.kernel_matrix = kernel_matrix
                kernel_matrix = pd.DataFrame(kernel_matrix)
                st.write(kernel_matrix)
        if st.session_state.kernel_matrix is not None:
            with st.form(key='save_form'):
                filenamek = st.text_input("Enter filename to save the kernel matrix", state["file_name"]+"_" + kernel_type + "_" + "kernel_matrix.csv")
                save_button = st.form_submit_button(label='Save Kernel Matrix')
            if save_button:
                if file_exists(filenamek):
                    st.error(f"File {filenamek} already exists. Please choose a different name.")
                else:
                    filenamek = os.path.join("newdata", filenamek)
                    save_kernel_matrix(st.session_state.kernel_matrix, filenamek)
                    st.success(f"Kernel matrix saved to {filenamek}.")
    pass
