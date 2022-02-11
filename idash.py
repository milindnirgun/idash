from this import d
import streamlit as st
import mplfinance as mpf
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt

@st.cache(suppress_st_warning=True)
def get_yahoo_data(ticker: str, start_date:dt.date=None) -> pd.DataFrame:
    """This function calls the yahoo API to download historical data for the given ticker. It uses a
    start_date if passed or defaults to whatever earliest the API returns.

    Args:
        ticker (str): A valid Yahoo symbol
        start_date (dt.datetime.date): A datetime.date value
    Returns:
        pd.DataFrame: historical data with date as the index
    """
    print("calling API with ", ticker)
    df = pdr.get_data_yahoo(ticker, start=start_date, end=dt.datetime.today())
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    #if start_date:
    #    df = df[df.index >= start_date]
    print("earliest date: ", df.index.min())

    return df


def create_ticker_dict(ticker: str, title: str) -> dict:
    """Sets up a dictionary for each symbol with the ticker, historical price data since 1/1/2019 and a title
    that is descriptive name for the symbol. 

    Args:
        ticker (str): A valid Yahoo ticker
        title (str): Name of the ticker passed
    Returns:
        dict: a dictionary containing the symbol, historical data for it and a descriptive name
    """
    item = {}
    item['ticker'] = ticker
    item['data'] = get_yahoo_data(ticker, dt.date(2019, 1, 1))
    item['title'] = title
    return item

def create_indices_list() -> list:
    """This is main method called for each run to create a list of dictionaries for each index or stock. This
    is hardcoded now for specific indices.

    Returns:
        list: a list of dictionaries for each symbol
    """
    idx_list = []
    d1 = create_ticker_dict('^DJI', 'Dow Jones Industrial')
    d2 = create_ticker_dict('^GSPC', 'S & P 500')
    d3 = create_ticker_dict('^IXIC', 'NASDAQ Composite')
    d4 = create_ticker_dict('^VIX', 'Volatility Index')

    return [d1, d2, d3, d4]

# Returns the number of months for a period. The Period value passed is from the slider in the form of string. 
# This is converted to an equivalent integer as number of months eg. 3mo = 3, 1y = 12
def get_months(period: str) -> int:
    """Returns the number of months for a period. The Period value passed is from the slider in the form of string. 
    This is converted to an equivalent integer as number of months eg. 3mo = 3, 1y = 12

    Args:
        period (str): a string which is output of the st.slider() denoting the period selected by user
    
    Returns:
        int: numerical value in months for the period passed
    """
    # In absence of switch case in python (<3.10), use if-elif-else
    if period == '3mo':
        months = 3
    elif period == '6mo':
        months = 6
    elif period == '1y':
        months = 12
    elif period == '2y':
        months = 24

    return months

@st.cache
def get_data_for_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Extract rows by range of dates in the past from today based on the period passed.
    #TBD: Add validation logic to ensure period does not exceed total data

    Args:
        symbol_list (list): list of dict objects representing symbols, data, name
        period (str): time duration selected by user from the slider

    Returns:
        pd.DataFrame: returns the data for the selected period
    """
    months = get_months(period)
    start_day_of_period = dt.date.today().replace(day=1) - dt.timedelta(days=months*30)
    start_day_of_period = start_day_of_period.replace(day=1)
    df_range = df.loc[start_day_of_period:dt.date.today()]
    
    return df_range


def get_plot(data: pd.DataFrame, apdict, title: str, chart_type: str, chart_style: str):
    """Creates a mplfinance plot with the given parameters.

    Args:
        data (pd.DataFrame): dataframe containing historical price & volume data for the symbol
        apdict (dict): contains additional plots to add (custom moving averages)
        title (str): the name of the symbol being plotted
        chart_type (str): type of chart selected from options
        chart_style (str): one of the selected options for style
    
    Returns:
        matplotlib.figure.Figure: Figure class from matplotlib 
        list: the axes objects from matplotlib
    """

    cs = mpf.make_mpf_style(base_mpf_style=chart_style, rc={'font.size':20, 'xtick.labelsize':14, 'ytick.labelsize':14})

    return mpf.plot(data,
                addplot=apdict,
                volume=True,
                #figsize=(15,10),
                figscale=1.5,
                figratio=(18, 10),
                title=title,
                type=chart_type,
                style=cs,
                tight_layout=True,
                ylabel="Price",
                returnfig=True )


def set_mav_legend(axlist: list, ma_tup: tuple):
    """Sets the legends for the Moving Averages lines. The first line is the main plot line for the ticker hence we
    trick by skipping it. To do that we get the list of all lines and pass [1:] to legend().

    Args:
        axlist (list): list of lines from the plot
        ma_tup (tuple): tuple containing the no. of days in MAV

    """
    # format the legend for the given MAs
    ma_legend = [str(m) + ' day' for m in ma_tup]

    # get the list of lines so we can print legends for only the MAV by filtering out the main chart line
    l = axlist[0].get_lines()
    axlist[0].legend(l[1:], ma_legend)


def set_xaxis_ticks(axlist: list, df: pd.DataFrame):
    """Sets the number of ticks for the X Axis for more control of overriding the default. Also sets the formatted labels
    and other aesthetics.

    Args:
        axlist (list): list of lines from the plot
        df (pd.DataFrame): dataframe containing historical price & volume data for the symbol

    """
    min_date = df.index.min()
    max_date = df.index.max()
    delta = max_date - min_date
    if (delta.days <= 72):
        tick_interval = '5d'
    elif (delta.days <= 200):
        tick_interval = '10d'
    elif (delta.days <= 400):
        tick_interval = '20d'
    else:
        tick_interval = '30d'
    ticks = pd.date_range(min_date, max_date, freq=tick_interval)
    # determine the tick locations from the data to be plotted based on ticks calculated on the range)
    ticklocations = [ df.index.get_loc(tick, method='nearest') for tick in ticks ]
    ticklabels = [ tick.date().strftime('%m/%d/%y') for tick in ticks]
    
    axlist[0].xaxis.set_ticks(ticklocations)
    axlist[0].set_xticklabels(ticklabels)
    #axlist[0].tick_params(axis='both', which='major', labelsize=14)

def set_custom_mav(df: pd.DataFrame, mavs: tuple, num_rows: int):
    """Calculates moving averages for the data passed. The raison d'etre for this function inside of using the 
    built-in function of mplfinance is to calculate the MAV for the entire data range available instead of just
    the period selected. This gives a more continuous MAV line for the entire chart. The resulting series is
    then passed to mplfinance.make_addplot() to create plot lines for each average.

    Args:
        df (pd.DataFrame): dataframe containing historical price & volume data for the maximum range
        mavs (tuple): tuple of integer values for each MA to be calculated and plotted
        num_rows (int): the number of values to be extracted from the calculated series to add to the plot. This
            is based on the period selected for plotting to match the plot data on the X axis
    
    Returns:
        list: list of additional plot objects for the MAVs
    """

    apd = []

    for mav in mavs:
        # Calculate SMA manually for the full range to plot continuously
        m = df['Close'].rolling(mav).mean()
        m = m.tail(num_rows)     # get only the last num_rows to match the number of datapoints to be plotted
        apd.append(mpf.make_addplot(m))

    return apd


# Main function to draw the chart page with all options
def main():
    print("Streamlit: ", st.__version__)
    print("mplfinance: ", mpf.__version__)
    print("pandas: ", pd.__version__)

    st.title("US Indices Dashboard")

    # TODO - Create a Form in sidebar for chartstyle, type, add MAV
    st.sidebar.subheader('Settings')
    st.sidebar.caption('Adjust charts settings and then press apply')

    # Set default values from session state
    if 'default_style' not in st.session_state:
        st.session_state.default_style = 'binance'

    with st.sidebar.form('settings_form'):
        # https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb
        chart_styles = [ 'binance', 'classic', 'yahoo', 'nightclouds' ]
        chart_style = st.selectbox('Chart style', options=chart_styles, index=chart_styles.index(st.session_state.default_style))
        
        chart_types = [ 'candle', 'ohlc', 'line', 'renko', 'pnf' ]
        chart_type = st.selectbox('Chart type', options=chart_types, index=chart_types.index('line'))

        mav1 = st.number_input('Mav 1', min_value=5, max_value=21, value=11, step=1)
        mav2 = st.number_input('Mav 2', min_value=20, max_value=60, value=50, step=5)
        mav3 = st.number_input('Mav 3', min_value=50, max_value=200, value=200, step=10)
        st.form_submit_button('Apply')

    mav_tuple = (int(mav1), int(mav2), int(mav3))
    st.session_state.default_style = chart_style

    period = st.select_slider('Range',
                options=['3mo','6mo','1y','2y'],
                value='6mo',
                help="Use the slider to select the chart duration along the X axis")

    # get Indices data
    ilist = create_indices_list()
    
    # Iterate for all tickers in the list
    for i in ilist:
        data = get_data_for_period(i['data'], period)
        no_rows = len(data.index)
        # Get list of MAVs for given tuple of values
        apds = set_custom_mav(i['data'], mav_tuple, no_rows)
        fig, ax = get_plot(data, apds, i['title'], chart_type, chart_style)
        set_mav_legend(ax, mav_tuple)
        set_xaxis_ticks(ax, data)

        # Plot the final chart
        st.pyplot(fig)

    st.caption("Data source - Yahoo Finance")
    
if __name__ == '__main__':
    main()
