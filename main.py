# -*- coding: utf-8 -*-
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import streamlit as st
from PIL import Image

from util.functions.path import get_file_path, get_dir_name, util_str, data_str
from omic_learn_2 import *
from util.pages.home_page import home_page
# from util.pages.overview_page import overview_page
# from util.pages.pdb_page import pdb_page
# from util.pages.conformation_page import conformation_page
# from util.pages.mutation_page import mutation_page
# from util.pages.inhibitor_page import inhibitor_page
# from util.pages.query_page import query_page
# from util.pages.classify_page import classify_page


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        # img = Image.open(
        #     get_file_path(
        #         "rascore_logo.png",
        #         dir_path=f"{get_dir_name(__file__)}/{util_str}/{data_str}",
        #     ),
        # )
        #
        # st.set_page_config(page_title="rascore", page_icon=img, layout="wide")

        st.sidebar.markdown("## Main Menu")
        app = st.sidebar.selectbox(
            "Select Page", self.apps, format_func=lambda app: app["title"]
        )
        st.sidebar.markdown("---")
        app["function"]()


app = MultiApp()

app.add_app("Home Page", home_page)
app.add_app("Cross-Validation", OmicLearn_Main)
# app.add_app("Data Processing", OmicLearn_Main1)
app.add_app("Compare to The-Staet-of-Art", OmicLearn_Main3)
app.add_app("loocv", OmicLearn_Main4)
app.add_app("Kernel", OmicLearn_Main5)
# app.add_app("Database Overview", overview_page)
# app.add_app("Query Database", query_page)
# app.add_app("Classify Structures", classify_page)
if __name__ == '__main__':

    app.run()
