from dash import dcc
from dash import html
from dash import dash_table
import dash_bootstrap_components as dbc
import datetime
from app import app
import plotly.graph_objects as go


empty_layout = go.Layout(width = 500,
                         height = 250,
                         showlegend=False,
                         xaxis=dict(autorange=True,
                                    showgrid=False,
                                    zeroline=False,
                                    showline=False,
                                    ticks='',
                                    showticklabels=False),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False
    ),
    annotations=[
        dict(
            x=2,
            y=2,
            xref='x',
            yref='y',
            text='Click "Calculate Arrival" to Generate Graph',
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=0
        )
    ]
)

result_table = dash_table.DataTable(
    id='result-table',
    page_size = 10,
    style_as_list_view = True,
    style_table = {
        'height': '100%',
        'width' : '100%',
        'padding-left' : 0,
        'padding-right' : 0,
        'padding-top': 0,
        'border' : '1px solid black'},
    style_header = {'textAlign' : 'left',
                    'fontWeight': 'bold',
                    'backgroundColor' : '#3B3331',
                    'color' : 'white'},
    style_cell = {'minWidth' : '1px',
                  'padding-left' : 25,
                  'padding-right' : 50,
                  'textAlign' : 'left',
                  'width' : '10px',
                  'maxWidth' : '25px',
                  'overflow' : 'hidden',
                  'textOverflow' : 'ellipsis',
                  'backgroundColor' : '#F4F0EF',
                  'color' : 'black'
                  }
)


summary = html.Div([

    html.H2('Stephen Brock | Flight Delay Simulation', style={'padding-left': 25, 'padding-top': 10,
            'padding-bottom': 5, 'background-color': '#FF0000', 'color': 'white', 'border-bottom': 'black'}),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Row([html.P("Departure Airport:")]),
                dbc.Row([dcc.Dropdown(id='departure-select', options=[{'label': 'BWI', 'value': 'BWI'}],
                                      clearable=False, value='BWI')])
            ], width=2),
            dbc.Col([
                dbc.Row([html.P("Destination Airport:")]),
                dbc.Row([dcc.Dropdown(id='destination-select', options=[{'label': 'JFK', 'value': 'JFK'},{'label': 'LAX', 'value': 'LAX'}],
                                      clearable=False, value='JFK')]),
            ], width=2)
        ], style={'padding-top': '10px'}),

        dbc.Row(
            dbc.Tabs([
                #Begin Airport Capacity Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Departure Delays"),
                                dbc.CardBody([
                                    dcc.Loading(dcc.Graph(id='gate-departure-delay-chart', config={'displayModeBar': False},
                                                          style={'float': 'left'})),
                                    dcc.Loading(dcc.Graph(id='taxi-departure-delay-chart', config={'displayModeBar': False},
                                                          style={'float': 'right'}))
                                ])
                            ],style = {'height' : '350px'})
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Flight Duration"),
                                dbc.CardBody([
                                    dcc.Loading(dcc.Graph(id='flight-duration-chart', config={'displayModeBar': False})),
                                ])
                            ],style = {'height' : '350px'})
                        ], width=3),

                    ], style={'padding-top': '10px'}),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Arrival Delays"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        dcc.Graph(id='gate-destination-delay-chart', config={'displayModeBar': False},
                                                  style={'float': 'left'})),
                                    dcc.Loading(
                                        dcc.Graph(id='taxi-destination-delay-chart', config={'displayModeBar': False},
                                                  style={'float': 'right'}))
                                ])
                            ],style = {'height' : '350px'})
                        ], width=8),
                    ], style={'margin-top': '10px'}),
                ], label='Delay Analysis'),
                #Begin Simulation Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.DatePickerSingle(id='travel-date-picker', with_portal=True, date=datetime.date.today(),
                                                 min_date_allowed=datetime.date.today())
                        ], width=1),
                        dbc.Col([
                            dcc.Dropdown(id='travel-hour-picker',
                                         options=["6:00 am", "7:00 am", "8:00 am", "9:00 am", "10:00 am", "11:00 am",
                                                  "12:00 pm","1:00 pm", "2:00 pm", "3:00 pm", "4:00 pm", "5:00 pm","6:00 pm", "7:00 pm", "8:00 pm", "9:00 pm", "10:00 pm"],
                                         clearable=False, value='6:00 am')
                        ], width=1),
                        dbc.Col([
                            html.Button('Calculate Arrival', id='run-sim-button', n_clicks=0),
                        ], width=1),
                    ], style={'padding-top': '10px'}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Arrival Estimate"),
                                dbc.CardBody(
                                    dcc.Loading([result_table]))
                            ],style = {'height' : '350px'})
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Simulation Results"),
                                dbc.CardBody(dcc.Loading(dcc.Graph(id='mean-wait-hist',figure = go.Figure(layout = empty_layout),config={'displayModeBar': False})))
                            ],style = {'height' : '350px'})
                        ], width=4),
                    ], style={'padding-top': '10px', 'padding-bottom': '10px'})
                ], label='Flight Time Simulation'),
            ], style={'margin-left': '10px', 'padding-top': '10px'})
        ),
    ], fluid=True)
])








