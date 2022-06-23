import pandas as pd


def linear_interpolation(data: pd.DataFrame, config):
    if config.time == 'daily':
        data['time'] = pd.to_datetime(data['time'], format=r'%Y/%m/%d')
        data = data.set_index('time')
        data = data.resample('D')
        data = data.interpolate(method=config.interpolation)
        data.reset_index(inplace=True)

    elif config.time == 'monthly':
        data['time'] = pd.to_datetime(data['time'], format=r'%Y/%m/%d')
        data = data.set_index('time')
        data = data.resample('M')
        data = data.interpolate(method=config.interpolation)
        data.reset_index(inplace=True)

    else:
        data = None

    return data
