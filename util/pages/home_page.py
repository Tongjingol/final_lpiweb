# -*- coding: utf-8 -*-

import streamlit as st
from PIL import Image

from ..functions.table import mask_equal
from ..functions.col import pdb_code_col
from ..functions.path import pages_str, data_str, get_file_path, load_table
from ..functions.gui import load_st_table, write_st_end, create_st_button, show_st_structure, get_neighbor_path


def show_structure():
    left_col, right_col = st.columns(2)

    df = load_st_table(__file__)

    dfm = load_table('.\\util\\data\\entry.tsv')
    show_st_structure(mask_equal(dfm, pdb_code_col, "6oim"),  # 6oim
                      zoom=1.2,
                      width=300,
                      height=250,
                      cartoon_trans=0,
                      surface_trans=1,
                      spin_on=True,
                      st_col=left_col)


def home_page():
    left_col, right_col = st.columns(2)

    df = load_st_table(__file__)

    dfm = load_table('.\\util\\data\\entry.tsv')
    show_st_structure(mask_equal(dfm, pdb_code_col, "6oim"),  # 6oim
                      zoom=1.2,
                      width=300,
                      height=250,
                      cartoon_trans=0,
                      surface_trans=1,
                      spin_on=True,
                      st_col=left_col)

    right_col.markdown("# View LPI-KCGCN")
    right_col.markdown("### A visual interface for predicting lncRNA and protein")
    right_col.markdown("**Created by Dongdong Mao**")
    right_col.markdown("**Tianjin University of Technology**")

    database_link_dict = {
        "NPInter v4.0": "http://bigdata.ibp.ac.cn/npinter4/",
        "Noncode": "http://www.noncode.org/",
        "SUPERFAMILY ": "https://supfam.org/SUPERFAMILY/",
    }

    st.sidebar.markdown("## Database-Related Links")
    for link_text, link_url in database_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)

    community_link_dict = {
        "NCI RAS Initiative": "https://www.cancer.gov/research/key-initiatives/ras",
        "KRAS Kickers": "https://www.kraskickers.org",
        "RASopathies Network": "https://rasopathiesnet.org",
    }

    st.sidebar.markdown("## Community-Related Links")
    for link_text, link_url in community_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)

    software_link_dict = {
        "BioPython": "https://biopython.org",
        "RDKit": "https://www.rdkit.org",
        "PDBrenum": "http://dunbrack.fccc.edu/PDBrenum/",
        "Fpocket": "https://bioserv.rpbs.univ-paris-diderot.fr/services/fpocket/",
        "PyMOL": "https://pymol.org/2/",
        "3Dmol": "https://3dmol.csb.pitt.edu",
        "Pandas": "https://pandas.pydata.org",
        "NumPy": "https://numpy.org",
        "SciPy": "https://scipy.org",
        "Sklearn": "https://scikit-learn.org/stable/",
        "Matplotlib": "https://matplotlib.org",
        "Seaborn": "https://seaborn.pydata.org",
        "Streamlit": "https://streamlit.io",
    }

    st.sidebar.markdown("## Software-Related Links")
    link_1_col, link_2_col, link_3_col = st.sidebar.columns(3)

    i = 0
    link_col_dict = {0: link_1_col, 1: link_2_col, 2: link_3_col}
    for link_text, link_url in software_link_dict.items():

        st_col = link_col_dict[i]
        i += 1
        if i == len(link_col_dict.keys()):
            i = 0

        create_st_button(link_text, link_url, st_col=st_col)

    st.markdown("---")

    st.markdown(
        """
        ### Summary
        *View LPI-KCGCN* is a visualization tool for our paper (Prediction of LncRNA-Protein Interactions
         Based on Kernel Combinations and Graph Convolutional Networks-(IEEE JBHI)). The *View LPI-KCGCN* 
        Mainly used to visualize our algorithmic framework LPI-KCGCN.Users can adjust parameters in real-time and 
        visualize the results.

        """
    )

    left_col, right_col = st.columns(2)

    img = Image.open(
        get_file_path(
            "flow.png",
            dir_path='.\\src\\rascore\\util\\data',
        )
    )

    left_col.image(img, output_format="PNG")

    left_col.markdown(
        """
        ### Usage

        To the up, is a dropdown main menu for navigating to 
        each page in the *View LPI-KCGCN*:

        - **Home Page:** We are here!
        - **Cross-Validation:** This part of the function is mainly used to achieve cross validation, and users can see the running results in real-time. This allows for arbitrary adjustment of model parameters and the ability to draw ROC and PR curves.
        - **Compare to The-Staet-of-Art:** This section mainly compares the prediction results of our prediction framework LPI-KCGCN with other methods. We use a three-dimensional surface graph to visualize the data.
        - **loocv:** This part of the function mainly corresponds to the case study in the paper, which allows for arbitrary selection of an lncRNA or a protein number, and predicts the interaction strength of the corresponding protein or lncRNAs using the LOOCV method. We also visualized this part of the content.
        """
    )
    st.markdown("---")

    # left_info_col, right_info_col = st.columns(2)
    #
    # left_info_col.markdown(
    #     f"""
    #     ### Authors
    #     Please feel free to contact us with any issues, comments, or questions.
    #
    #     ##### Mitchell Parker [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/bukotsunikki.svg?style=social&label=Follow%20%40Mitch_P)](https://twitter.com/Mitch_P)
    #
    #     - Email:  <mip34@drexel.edu> or <mitchell.parker@fccc.edu>
    #     - GitHub: https://github.com/mitch-parker
    #
    #     ##### Roland Dunbrack [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/bukotsunikki.svg?style=social&label=Follow%20%40RolandDunbrack)](https://twitter.com/RolandDunbrack)
    #
    #     - Email: <roland.dunbrack@fccc.edu>
    #     - GitHub: https://github.com/DunbrackLab
    #     """,
    #     unsafe_allow_html=True,
    # )
    #
    # right_info_col.markdown(
    #     """
    #     ### Funding
    #
    #     - NIH NIGMS F30 GM142263 (to M.P.)
    #     - NIH NIGMS R35 GM122517 (to R.D.)
    #      """
    # )
    #
    # right_info_col.markdown(
    #     """
    #     ### License
    #     Apache License 2.0
    #     """
    # )

    write_st_end()
