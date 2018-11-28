#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 03:01:15 2018

@author: budi
"""
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import pandas as pd

import plotly
import dash_table_experiments as dt



def generate_table(dataframe):
    print('html.Table(className='mytable',children= \
        [html.Thead(html.Tr([html.Th(col) for col in dataframe.columns]))] +
        [html.Tbody(html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe)))])

df = pd.read_excel('tweets.xlsx')