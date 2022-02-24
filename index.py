from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from apps.func import mean_wait_hist, taxi_delay_chart, gate_delay_chart, flight_duration_chart, get_dist_parameters
from app import app
from apps.layouts import summary
from dash.exceptions import PreventUpdate
from app import server
from apps.query import queryData


dataset_df = queryData()
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='simulation-parameters')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/summary' or pathname == '/summary/':
        return summary
    else:
        return summary



#airport capacity tab callback
@app.callback([Output("gate-departure-delay-chart", 'figure'),
               Output("taxi-departure-delay-chart", 'figure'),
               Output('flight-duration-chart','figure'),
               Output("gate-destination-delay-chart", 'figure'),
               Output("taxi-destination-delay-chart", 'figure'),
               Output("simulation-parameters", "data")],
              [Input("departure-select", "value"),
               Input("destination-select", "value")]
)
def update_delay_tab(departure_airport, destination_airport):

    data_df = dataset_df.copy()


    departure_df = data_df[data_df['ORIGIN'] == departure_airport]
    destination_df = data_df[data_df['DEST'] == destination_airport]
    flight_df = data_df[(data_df['ORIGIN'] == departure_airport) & (data_df['DEST'] == destination_airport)]


    departure_gate_graph, destination_gate_graph = gate_delay_chart(departure_df, destination_df)
    departure_taxi_graph, destination_taxi_graph = taxi_delay_chart(departure_df, destination_df)
    flight_duration_graph = flight_duration_chart(flight_df)

    distribution_data = get_dist_parameters(departure_df, destination_df, flight_df)

    distribution_data=distribution_data

    return departure_gate_graph, departure_taxi_graph, flight_duration_graph, destination_gate_graph, destination_taxi_graph, distribution_data




#simulation tab callback
@app.callback([Output("mean-wait-hist", 'figure'),
               Output("result-table", 'data'),
               Output("result-table", 'page_current')],
              [Input("run-sim-button", "n_clicks")],
              [State('travel-hour-picker','value'),
               State('simulation-parameters', 'data')]
)
def run_sim(n_clicks,flight_begin_time, pdf_params):
    if n_clicks == 0:
        raise PreventUpdate


    mean_hist,result_df = mean_wait_hist(flight_begin_time,pdf_params)



    return mean_hist, result_df.to_dict('records'), 0


if __name__ == '__main__':
    app.run_server(debug=False)
