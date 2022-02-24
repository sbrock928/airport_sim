import pandas as pd
from app import app
from datetime import datetime, timedelta
import numpy as np
from apps.simulation import Airport_Simulation
from scipy import stats
import plotly.graph_objects as go

graph_height = 250
graph_width = 500





def get_dist_parameters(temp_departure_df, temp_destination_df, temp_flight_df):
    taxi_out_df = temp_departure_df
    gate_out_df = temp_departure_df
    taxi_in_df = temp_destination_df
    gate_in_df = temp_destination_df
    ref_flight_df = temp_flight_df
    departure_rate_df = temp_departure_df
    arrival_rate_df = temp_destination_df

    attribute_dict = {}


    '---Calculate Taxi Delay PDF Parameters---'
    'remove extreme values from dataset'
    z1 = np.abs(stats.zscore(taxi_out_df['TAXI_OUT'].values))
    z2 = np.abs(stats.zscore(taxi_in_df['TAXI_IN'].values))
    taxi_out_df= taxi_out_df[(z1 < 3)]
    taxi_in_df= taxi_in_df[(z2 < 3)]

    dist = getattr(stats, 'lognorm')
    shape, loc, scale = dist.fit(taxi_out_df['TAXI_OUT'])
    mu = np.log(scale)
    sigma = shape
    attribute_dict['taxi_out'] = {'mu' : mu, 'sigma' : sigma}

    shape, loc, scale = dist.fit(taxi_in_df['TAXI_IN'])
    mu = np.log(scale)
    sigma = shape
    attribute_dict['taxi_in'] = {'mu' : mu, 'sigma' : sigma}


    '---Calculate Gate Delay PDF Parameters---'
    'remove extreme values from dataset'
    gate_out_df = gate_out_df[(gate_out_df['DEP_DELAY']>0) & (gate_out_df['DEP_DELAY']<60)]
    gate_in_df = gate_in_df[(gate_in_df['ARR_DELAY']>0) & (gate_in_df['ARR_DELAY']<60)]

    z1 = np.abs(stats.zscore(gate_out_df['DEP_DELAY'].values))
    z2 = np.abs(stats.zscore(gate_in_df['ARR_DELAY'].values))
    gate_out_df= gate_out_df[(z1 < 3)]
    gate_in_df= gate_in_df[(z2 < 3)]

    dist = getattr(stats, 'expon')
    loc, scale= dist.fit(gate_out_df['DEP_DELAY'])
    attribute_dict['gate_out'] = {'loc': loc, 'scale': scale}

    loc, scale= dist.fit(gate_in_df['ARR_DELAY'])
    attribute_dict['gate_in'] = {'loc': loc, 'scale': scale}




    '---Calculate Flight PDF Parameters---'
    ref_flight_df = ref_flight_df[ref_flight_df['AIR_TIME'] > 0]
    z = np.abs(stats.zscore(ref_flight_df['AIR_TIME'].values))
    ref_flight_df=ref_flight_df[(z < 3)]

    dist = getattr(stats, 'lognorm')
    shape, loc, scale = dist.fit(ref_flight_df['AIR_TIME'],floc=0)
    mu = np.log(scale)
    sigma = shape
    attribute_dict['flight_time'] = {'mu' : mu, 'sigma' : sigma}



    '---Calculate Arrival/Departure Rate---'

    departure_rate_df = departure_rate_df.set_index('FL_DATE')
    departure_rate_df = departure_rate_df.sort_values(by=['FL_DATE','DEP_TIME'], ascending=True)
    departure_rate_df['dep_timediff'] = departure_rate_df['DEP_TIME'].diff(1).dt.total_seconds().div(60)
    departure_rate_df.dropna(inplace=True)
    departure_rate_df = departure_rate_df[departure_rate_df['dep_timediff']>=0]

    arrival_rate_df = arrival_rate_df.set_index('FL_DATE')
    arrival_rate_df = arrival_rate_df.sort_values(by=['FL_DATE','ARR_TIME'], ascending=True)
    arrival_rate_df['arr_timediff'] = arrival_rate_df['ARR_TIME'].diff(1).dt.total_seconds().div(60)
    arrival_rate_df.dropna(inplace=True)
    arrival_rate_df = arrival_rate_df[arrival_rate_df['arr_timediff']>=0]


    dist = getattr(stats, 'expon')
    loc, scale= dist.fit(departure_rate_df['dep_timediff'])

    attribute_dict['dep_rate'] = {'loc': loc, 'scale': scale}


    loc, scale= dist.fit(arrival_rate_df['arr_timediff'])
    attribute_dict['arr_rate'] = {'loc': loc, 'scale': scale}


    return attribute_dict


def gate_delay_chart(temp_departure_df, temp_destination_df):

    gate_out_df = temp_departure_df
    gate_in_df = temp_destination_df


    gate_out_df = gate_out_df[(gate_out_df['DEP_DELAY']>0) & (gate_out_df['DEP_DELAY']<60)]
    gate_in_df = gate_in_df[(gate_in_df['ARR_DELAY']>0) & (gate_in_df['ARR_DELAY']<60)]


    'remove extreme values from dataset'
    z1 = np.abs(stats.zscore(gate_out_df['DEP_DELAY'].values))
    z2 = np.abs(stats.zscore(gate_in_df['ARR_DELAY'].values))
    gate_out_df= gate_out_df[(z1 < 3)]
    gate_in_df= gate_in_df[(z2 < 3)]


    gate_out_fig  = go.Figure(data=[go.Histogram(x=gate_out_df['DEP_DELAY'].values, xbins=dict(start=0,end=70,size=1), name='Data')])


    dist = getattr(stats, 'expon')
    loc, scale= dist.fit(gate_out_df['DEP_DELAY'])

    X_expon = np.random.exponential(scale=scale, size=len(gate_out_df))

    counts, bins = np.histogram(X_expon, bins=range(0, 70, 1))
    bins = bins[:-1] + (bins[1] - bins[0])
    gate_out_fig.add_trace(go.Scatter(x=bins, y=counts, mode='lines',name = 'Expon Fit'))

    gate_out_fig.update_layout(
        height = graph_height,
        width = graph_width,
        title_text='Gate Departure',  # title of plot
        xaxis_title_text='Delay Time',
        yaxis_title_text='Count',
        margin=dict(l=0, r=0, t=25, b=0, pad=5),
    )



    gate_in_fig  = go.Figure(data=[go.Histogram(x=gate_in_df['ARR_DELAY'].values, xbins=dict(start=0,end=70,size=1), name='Data')])

    dist = getattr(stats, 'expon')
    loc, scale= dist.fit(gate_in_df['ARR_DELAY'])
    X_expon = np.random.exponential(scale=scale, size=len(gate_in_df))

    counts, bins = np.histogram(X_expon, bins=range(0, 70, 1))
    bins = bins[:-1] + (bins[1] - bins[0])
    gate_in_fig.add_trace(go.Scatter(x=bins, y=counts, mode='lines',name = 'Expon Fit'))

    gate_in_fig.update_layout(
        height = graph_height,
        width = graph_width,
        title_text='Gate Arrival',  # title of plot
        xaxis_title_text='Delay Time',
        yaxis_title_text='Count',
        margin=dict(l=0, r=0, t=25, b=0, pad=5),
    )
    gate_in_fig.update_xaxes(
        range=[-1,100],  # sets the range of xaxis
        constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    )



    return gate_out_fig, gate_in_fig


def taxi_delay_chart(temp_departure_df, temp_destination_df):
    taxi_out_df = temp_departure_df
    taxi_in_df = temp_destination_df


    'remove extreme values from dataset'
    z1 = np.abs(stats.zscore(taxi_out_df['TAXI_OUT'].values))
    z2 = np.abs(stats.zscore(taxi_in_df['TAXI_IN'].values))
    taxi_out_df= taxi_out_df[(z1 < 3)]
    taxi_in_df= taxi_in_df[(z2 < 3)]



    'Departure Taxi Chart'
    taxi_out_fig  = go.Figure(data=[go.Histogram(x=taxi_out_df['TAXI_OUT'], xbins=dict(start=0,end=100,size=1), name='Data')])
    dist = getattr(stats, 'lognorm')
    shape, loc, scale = dist.fit(taxi_out_df['TAXI_OUT'])
    mu = np.log(scale)
    sigma = shape
    X_lognorm = np.random.lognormal(mean=mu, sigma=sigma, size=len(taxi_out_df))

    counts, bins = np.histogram(X_lognorm, bins=range(0, 100, 1))
    bins = bins[:-1] + (bins[1] - bins[0])
    taxi_out_fig.add_trace(go.Scatter(x=bins, y=counts, mode='lines',name = 'Lognorm Fit'))
    taxi_out_fig.update_layout(
        height = graph_height,
        width = graph_width,
        title_text='Taxi Out Delay',  # title of plot
        xaxis_title_text='Delay Time',
        yaxis_title_text='Count',
        margin=dict(l=0, r=0, t=25, b=0, pad=5),
    )


    taxi_out_fig.update_xaxes(
        range=[-1, 50],  # sets the range of xaxis
        constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    )


    'Arrival Taxi Chart'
    taxi_in_fig  = go.Figure(data=[go.Histogram(x=taxi_in_df['TAXI_IN'], xbins=dict(start=0,end=100,size=1), name='Data')])
    dist = getattr(stats, 'lognorm')
    shape, loc, scale = dist.fit(taxi_in_df['TAXI_IN'])
    mu = np.log(scale)
    sigma = shape
    X_lognorm = np.random.lognormal(mean=mu, sigma=sigma, size=len(taxi_in_df))

    counts, bins = np.histogram(X_lognorm, bins=range(0, 100, 1))
    bins = bins[:-1] + (bins[1] - bins[0])
    taxi_in_fig.add_trace(go.Scatter(x=bins, y=counts, mode='lines',name = 'Lognorm Fit'))
    taxi_in_fig.update_layout(
        height = graph_height,
        width = graph_width,
        title_text='Taxi In Delay',  # title of plot
        xaxis_title_text='Delay Time',
        yaxis_title_text='Count',
        margin=dict(l=0, r=0, t=25, b=0, pad=5),
    )


    taxi_in_fig.update_xaxes(
        range=[-1, 50],  # sets the range of xaxis
        constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    )



    return taxi_out_fig,taxi_in_fig

def flight_duration_chart(temp_flight_df):

    ref_flight_df = temp_flight_df
    ref_flight_df = ref_flight_df[ref_flight_df['AIR_TIME'] > 0]
    z = np.abs(stats.zscore(ref_flight_df['AIR_TIME'].values))

    ref_flight_df=ref_flight_df[(z < 3)]


    flight_duration_fig  = go.Figure(data=[go.Histogram(x=ref_flight_df['AIR_TIME'], xbins=dict(start=min(ref_flight_df['AIR_TIME'])-5,end=max(ref_flight_df['AIR_TIME'])+5,size=1), name='Data')])
    dist = getattr(stats, 'lognorm')
    shape, loc, scale = dist.fit(ref_flight_df['AIR_TIME'],floc=0)
    mu = np.log(scale)
    sigma = shape
    X_lognorm = np.random.lognormal(mean=mu, sigma=sigma, size=len(ref_flight_df))

    counts, bins = np.histogram(X_lognorm, bins=range(int(min(ref_flight_df['AIR_TIME']))-5, int(max(ref_flight_df['AIR_TIME']))+5, 1))
    bins = bins[:-1] + (bins[1] - bins[0])
    flight_duration_fig.add_trace(go.Scatter(x=bins, y=counts, mode='lines',name = 'Lognorm Fit'))
    flight_duration_fig.update_layout(
        height = graph_height,
        width = 400,
        title_text='Flight Duration',  # title of plot
        xaxis_title_text='Time in Air',
        yaxis_title_text='Count',
        margin=dict(l=0, r=0, t=25, b=0, pad=5),
    )

    return flight_duration_fig



def mean_wait_hist(flight_start, pdf_parameter_dict):

    start_time = datetime.strptime(flight_start, '%I:%M %p')
    temp_df = run_sim(pdf_parameter_dict)


    fig1 = go.Figure(data=[go.Histogram(x=temp_df['Flight Time'], nbinsx=50)])

    fig1.update_layout(
        height = graph_height,
        width = 500,
        title_text='Flight Duration',  # title of plot
        xaxis_title_text='Time in Air',
        yaxis_title_text='Count',
        margin=dict(l=0, r=0, t=25, b=0, pad=5),
    )

    resultdict = { 'Status' : ['Scheduled Departure', 'Departure Gate Delay','Departure Taxi Delay','Flight Time','Arrival Taxi Delay','Arrival Gate Delay'],
                   'Duration (in minutes)' : [0,temp_df['Avg Departure Gate Time'].mean(),temp_df['Avg Departure Taxi Time'].mean(),temp_df['Flight Time'].mean(),
                                 temp_df['Avg Arrival Taxi Time'].mean(),temp_df['Avg Arrival Gate Time'].mean()]}

    result_df = pd.DataFrame(resultdict)
    result_df['Duration (in minutes)'] = result_df['Duration (in minutes)'].round(2)
    result_df['Time Elapsed'] = result_df['Duration (in minutes)'].cumsum().round(2)
    result_df['Time'] = (start_time + pd.to_timedelta(result_df['Time Elapsed'], unit='m')).dt.strftime('%I:%M %p')

    return fig1,result_df


def run_sim(pdf_parameter_dict):

    departure = Airport_Simulation(pdf_parameter_dict['dep_rate'],pdf_parameter_dict['taxi_out'],pdf_parameter_dict['gate_out'])
    df_departure = pd.DataFrame(columns=['Average Inter-arrival Time', 'Average Gate Time', 'Average Taxi Time','Utilization', 'Delayed Plane Count', 'Daily Total Wait Time'])

    arrival = Airport_Simulation(pdf_parameter_dict['arr_rate'],pdf_parameter_dict['taxi_in'],pdf_parameter_dict['gate_in'])
    df_arrival = pd.DataFrame(columns=['Average Inter-arrival Time', 'Average Gate Time',
                                'Average Taxi Time','Utilization', 'Delayed Plane Count', 'Daily Total Wait Time'])

    df_trial = pd.DataFrame(columns=['Avg Departure Gate Time', 'Avg Departure Taxi Time', 'Flight Time', 'Avg Arrival Taxi Time','Avg Arrival Gate Time', 'Total Transit Time' ])


    for i in range(100):
        np.random.seed(i)
        departure.__init__(pdf_parameter_dict['dep_rate'],pdf_parameter_dict['taxi_out'],pdf_parameter_dict['gate_out'])
        while departure.clock <= 1440:
            departure.time_adv()

        a= pd.Series([departure.clock / departure.num_arrivals, departure.gate_sum / departure.num_of_departures, departure.taxi_sum / departure.num_of_departures,
                       (departure.gate_sum + departure.taxi_sum)  / departure.clock, departure.number_in_queue, departure.total_wait_time],
                      index=df_departure.columns)
        df_departure = df_departure.append(a, ignore_index=True)


        #flight time for trial
        flight_time = (np.random.lognormal(mean= pdf_parameter_dict['flight_time']['mu'], sigma=pdf_parameter_dict['flight_time']['sigma'], size = 1)[0])

        arrival.__init__(pdf_parameter_dict['arr_rate'], pdf_parameter_dict['taxi_in'],pdf_parameter_dict['gate_in'])
        while arrival.clock <= 1440:
            arrival.time_adv()


        b = pd.Series([arrival.clock / arrival.num_arrivals, arrival.gate_sum / arrival.num_of_departures, arrival.taxi_sum / arrival.num_of_departures,
                       (arrival.gate_sum + arrival.taxi_sum)/ arrival.clock, arrival.number_in_queue, arrival.total_wait_time],
                      index=df_arrival.columns)
        df_arrival = df_arrival.append(b, ignore_index=True)


        c = pd.Series([departure.gate_sum / departure.num_of_departures,departure.taxi_sum / departure.num_of_departures,
                       flight_time,
                       arrival.taxi_sum / arrival.num_of_departures,arrival.gate_sum / arrival.num_of_departures,
                       (departure.gate_sum / departure.num_of_departures) + (departure.taxi_sum / departure.num_of_departures) + flight_time
                       +(arrival.taxi_sum / arrival.num_of_departures) + (arrival.gate_sum / arrival.num_of_departures)],index=df_trial.columns)
        df_trial = df_trial.append(c, ignore_index=True)


    df_trial.replace([np.inf, -np.inf], np.nan, inplace=True)


    df_trial.dropna(inplace=True)



    return df_trial