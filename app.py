#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 23:33:50 2018

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




app = dash.Dash()

colors = [{
    'background': 'rgb(0, 32, 76)',
    'text': '#B3C8E5'
},
{
        'background': '#fef0d9',
        'text': 'rgb(30, 30, 30)'
    }
]


app.layout = html.Div(
    style={'margin': 10},
    children=[
        html.Div(style={
            'position': 'absolute',
            'width': '100%',
            'height': '100%',
            'backgroundColor': colors[0]['background']
        }),
        html.Div(
            className='container',
            style={'color': colors[0]['text'], 'paddingTop': 60},
            children=[
                html.H1('Deteksi Spam Tweet pada Twitter dengan Metode Naive Bayes'),
                html.H4('Budi Santoso - Sekolah Tinggi teknik Surabaya'),
                html.Hr(),
                html.Div([
                    html.Div(className='two columns', children=[
                        html.Div(className = 'sidenav',children=[
                          html.Button(children='Dataset',className='button',id='loadDataAwal'),
                          html.Button('Data Cleaning',className='button',id='loadCleanedData'),
                          html.Button('Konversi Kata',className='button',id='loadConvertData'),
                          html.Button('Filter Stopword',className='button',id='loadStoppedWord'),
                          html.Button('Stemming',className='button',id='loadStemmed'),
                          html.Button('Training',className='button',id='loadTFSpam'),
                          html.Hr(),
                                
                        ]),
                        
                    ]),
                    html.Div(className='ten columns',
                             id='content', 
                             style={'color': colors[1]['text'],
                                    'height':'400px'
                                    }),
                    html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'}),
                    
                ])
            ]
        )
    ]
)

#app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

# Functions

def generate_table(dataframe):
    rows = []
    for i in range(len(dataframe)):
        row = []
        for col in dataframe.columns:
            value = dataframe.iloc[i][col]
            row.append(html.Td(value))
        rows.append(html.Tr(row))

    return html.Table(className='sctable',children=
        # Header
        [html.Thead([html.Tr([html.Th(col) for col in dataframe.columns])])] +

        # Body
        [html.Tbody(rows)])

@app.callback(Output('content', 'children'), [Input('loadDataAwal', 'n_clicks')])
def loadDataAwal(n_clicks):
    if n_clicks > 0:
        df = pd.read_excel('tweets.xlsx')
        return html.Div(style={'backgroundColor': 'white'},children=[
            html.Div(id='datatable-output'),
            generate_table(df)
        ])




if __name__ == '__main__':
    app.run_server(debug=True)