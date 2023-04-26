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

# Session state
import utils.session_states as session_states

# ML functionalities
from utils.ml_helper import perform_cross_validation, transform_dataset, calculate_cm

# Plotting
from utils.plot_helper import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_pr_curve_cv,
    plot_roc_curve_cv,
    perform_EDA,
)

# UI components and others func.
from utils.ui_helper import (
    main_components,
    get_system_report,
    save_sessions,
    load_data,
    main_text_and_data_upload,
    objdict,
    generate_sidebar_elements,
    get_download_link,
    generate_text,
    generate_footer_parts,
)

# Set the configs
APP_TITLE = "OmicLearn — ML platform for omics datasets"
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=Image.open("./utils/omic_learn.ico"),
    layout="centered",
    initial_sidebar_state="auto",
)
icon = Image.open("./utils/6OIM.png")
report = get_system_report()

# This needs to be here as it needs to be after setting ithe initial_sidebar_state
try:
    import xgboost
except ModuleNotFoundError:
    st.warning(
        "**WARNING:** Xgboost not installed. To use xgboost install using `conda install py-xgboost`"
    )


# Display results and plots
def classify_and_plot(state):
    state.bar = st.progress(0)
    # Cross-Validation
    st.markdown("Performing analysis and Running cross-validation")
    cv_results, cv_curves = perform_cross_validation(state)

    st.header("Cross-validation results")

    top_features = []
    # Feature importances from the classifier
    with st.expander("Feature importances from the classifier"):
        st.subheader("Feature importances from the classifier")
        if state.cv_method == "RepeatedStratifiedKFold":
            st.markdown(
                f"This is the average feature importance from all {state.cv_splits * state.cv_repeats} cross validation runs."
            )
        else:
            st.markdown(
                f"This is the average feature importance from all {state.cv_splits} cross validation runs."
            )

        if cv_curves["feature_importances_"] is not None:

            # Check whether all feature importance attributes are 0 or not
            if (
                    pd.DataFrame(cv_curves["feature_importances_"]).isin([0]).all().all()
                    == False
            ):
                p, feature_df, feature_df_wo_links = plot_feature_importance(
                    cv_curves["feature_importances_"]
                )
                st.plotly_chart(p, use_container_width=True)
                if p:
                    get_download_link(p, "clf_feature_importance.pdf")
                    get_download_link(p, "clf_feature_importance.svg")

                # Display `feature_df` with NCBI links
                st.subheader("Feature importances from classifier table")
                st.write(
                    feature_df.to_html(escape=False, index=False),
                    unsafe_allow_html=True,
                )
                get_download_link(feature_df_wo_links, "clf_feature_importances.csv")

                top_features = feature_df.index.to_list()
            else:
                st.info(
                    "All feature importance attribute are zero (0). The plot and table are not displayed."
                )
        else:
            st.info(
                "Feature importance attribute is not implemented for this classifier."
            )
    state["top_features"] = top_features
    # ROC-AUC
    with st.expander(
            "Receiver operating characteristic Curve and Precision-Recall Curve"
    ):
        st.subheader("Receiver operating characteristic")
        p = plot_roc_curve_cv(cv_curves["roc_curves_"])
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "roc_curve.pdf")
            get_download_link(p, "roc_curve.svg")

        # Precision-Recall Curve
        st.subheader("Precision-Recall Curve")
        st.markdown(
            "Precision-Recall (PR) Curve might be used for imbalanced datasets."
        )
        p = plot_pr_curve_cv(cv_curves["pr_curves_"], cv_results["class_ratio_test"])
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "pr_curve.pdf")
            get_download_link(p, "pr_curve.svg")

    # Confusion Matrix (CM)
    with st.expander("Confusion matrix"):
        names = ["CV_split {}".format(_ + 1) for _ in range(len(cv_curves["y_hats_"]))]
        names.insert(0, "Sum of all splits")
        p = plot_confusion_matrices(
            state.class_0, state.class_1, cv_curves["y_hats_"], names
        )
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "cm.pdf")
            get_download_link(p, "cm.svg")

        cm_results = [calculate_cm(*_)[1] for _ in cv_curves["y_hats_"]]

        cm_results = pd.DataFrame(cm_results, columns=["TPR", "FPR", "TNR", "FNR"])
        # (tpr, fpr, tnr, fnr)
        cm_results_ = cm_results.mean().to_frame()
        cm_results_.columns = ["Mean"]

        cm_results_["Std"] = cm_results.std()

        st.write("Average peformance for all splits:")
        st.write(cm_results_)

    # Results table
    with st.expander("Table for run results"):
        st.subheader(f"Run results for `{state.classifier}`")
        state["summary"] = pd.DataFrame(pd.DataFrame(cv_results).describe())
        st.write(state.summary)
        st.info(
            """
            **Info:** `Mean precision` and `Mean recall` values provided in the table above
            are calculated as the mean of all individual splits shown in the confusion matrix,
            not the "Sum of all splits" matrix.
            """
        )
        get_download_link(state.summary, "run_results.csv")

    if state.cohort_checkbox:
        st.header("Cohort comparison results")
        cohort_results, cohort_curves = perform_cross_validation(
            state, state.cohort_column
        )

        with st.expander(
                "Receiver operating characteristic Curve and Precision-Recall Curve"
        ):
            # ROC-AUC for Cohorts
            st.subheader("Receiver operating characteristic")
            p = plot_roc_curve_cv(
                cohort_curves["roc_curves_"], cohort_curves["cohort_combos"]
            )
            st.plotly_chart(p, use_container_width=True)
            if p:
                get_download_link(p, "roc_curve_cohort.pdf")
                get_download_link(p, "roc_curve_cohort.svg")

            # PR Curve for Cohorts
            st.subheader("Precision-Recall Curve")
            st.markdown(
                "Precision-Recall (PR) Curve might be used for imbalanced datasets."
            )
            p = plot_pr_curve_cv(
                cohort_curves["pr_curves_"],
                cohort_results["class_ratio_test"],
                cohort_curves["cohort_combos"],
            )
            st.plotly_chart(p, use_container_width=True)
            if p:
                get_download_link(p, "pr_curve_cohort.pdf")
                get_download_link(p, "pr_curve_cohort.svg")

        # Confusion Matrix (CM) for Cohorts
        with st.expander("Confusion matrix"):
            st.subheader("Confusion matrix")
            names = [
                "Train on {}, Test on {}".format(_[0], _[1])
                for _ in cohort_curves["cohort_combos"]
            ]
            names.insert(0, "Sum of cohort comparisons")

            p = plot_confusion_matrices(
                state.class_0, state.class_1, cohort_curves["y_hats_"], names
            )
            st.plotly_chart(p, use_container_width=True)
            if p:
                get_download_link(p, "cm_cohorts.pdf")
                get_download_link(p, "cm_cohorts.svg")

        with st.expander("Table for run results"):
            state["cohort_summary"] = pd.DataFrame(pd.DataFrame(cv_results).describe())
            st.write(state.cohort_summary)
            get_download_link(state.cohort_summary, "run_results_cohort.csv")

        state["cohort_combos"] = cohort_curves["cohort_combos"]
        state["cohort_results"] = cohort_results

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

    # st.info(
    #     f"**INFO:** Found {fivefoldcv(lpi)} missing values. "
    #     "Use missing value imputation or `xgboost` classifier."
    # )
    # Analysis Part
    # if len(state.df) > 0 and state.target_column == "":
    #     st.warning("**WARNING:** Select classification target from your data.")
    #
    # elif len(state.df) > 0 and not (state.class_0 and state.class_1):
    #     st.warning("**WARNING:** Define classes for the classification target.")
    # if st.button("Reconstruct Kernels", key="run"):

    if (
            # (state.df is not None)
            # and (state.class_0 and state.class_1) and
            (st.button("Run analysis", key="run"))
    ):
        # st.info(
        #         f"**INFO:** Found {fivefoldcv(lpi)} missing values. "
        #         "Use missing value imputation or `xgboost` classifier."
        #     )

        # state.features = state.proteins + state.additional_features
        # subset = state.df_sub[
        #     state.df_sub[state.target_column].isin(state.class_0)
        #     | state.df_sub[state.target_column].isin(state.class_1)
        # ].copy()
        # state.y = subset[state.target_column].isin(state.class_0)
        # state.X = transform_dataset(subset, state.additional_features, state.proteins)

        # if state.cohort_column is not None:
        #     state["X_cohort"] = subset[state.cohort_column]

        # Show the running info text
        st.info(
            f"""
            **Running info:**
            - Using the following features: **Class 0 ``, Class 1 ``**.
            - Using classifier **``**.
            - Using a total of  **``** features.
            - Note that OmicLearn is intended to be an exploratory tool to assess the performance of algorithms,
                rather than providing a classification model for production.
        """
        )

        # Plotting and Get the results
        # state = classify_and_plot(state)

        # Generate summary text
        # generate_text(state, report)

        # Session and Run info
        widget_values["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " (UTC)"
        if state.sample_file == "all" and state.sample_file_protein == "all":
            fivefoldcv(lpi)
        else:
            s5_main(state)
        #
        # for _ in state.summary.columns:
        #     widget_values[_ + "_mean"] = state.summary.loc["mean"][_]
        #     widget_values[_ + "_std"] = state.summary.loc["std"][_]
        #
        # user_name = str(random.randint(0, 10000)) + "OmicLearn"
        # session_state = session_states.get(user_name=user_name)
        # widget_values["user"] = session_state.user_name
        # widget_values["top_features"] = state.top_features
        # save_sessions(widget_values, session_state.user_name)

        # Generate footer
        generate_footer_parts(report)

    # else:
    #     pass


# Run the OmicLearn
# if __name__ == "__main__":
#     try:
#         OmicLearn_Main()
#     except (ValueError, IndexError) as val_ind_error:
#         st.error(
#             f"There is a problem with values/parameters or dataset due to {val_ind_error}."
#         )
#     except TypeError as e:
#         # st.warning("TypeError exists in {}".format(e))
#         pass
