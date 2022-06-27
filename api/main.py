from fastapi import FastAPI
from typing import Dict
from khayyam import JalaliDate
import uvicorn
from utils.datatypes import *
from utils.interpolation import linear_interpolation
import pandas as pd
from utils.outlier import isolation_forest, dbscan, lof

app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'Hello World!'}


@app.post('/interpolation/simple')
async def interpol(data: Dict, config: Config):
    df = pd.DataFrame.from_dict(data)
    df['time'] = pd.to_datetime(df['time'], format=r'%Y/%m/%d')
    if config.type == CalenderType.shamsi:
        result = linear_interpolation(df, config)
        result['time'] = [JalaliDate(time_.date()).strftime(r'%Y/%m/%d') for time_ in result['time']]
    else:
        result = linear_interpolation(df, config)

    return {'data': result.to_dict()}


@app.post('/interpolation/timeswitch')
async def tinterpol(data: Dict, config: ConfigTimeSwitch):
    df = pd.DataFrame.from_dict(data)
    df['time'] = pd.to_datetime(df['time'], format=r'%Y/%m/%d')
    result = linear_interpolation(df, config)

    return {'data': result.to_dict()}


@app.post('/detect/outlier')
async def detection(data: Dict, config: ConfigDetection):
    df = pd.DataFrame.from_dict(data)
    if config.timeseries:
        isf = isolation_forest(df)['outliers']
        lo = lof(df)['outliers']

        df['isolation_forest'] = isf
        df['lof'] = lo
    else:
        dbscan_ = dbscan(df)['outliers']
        isf = isolation_forest(df)['outliers']

        df['dbscan'] = dbscan_
        df['isolation_forest'] = isf

    data = data.to_dict()
    return {'data': data, 'config': config}


@app.post('/management/balanced')
async def balanced(data: Dict[str, float], config: ConfigBalanced):
    return {'data': data, 'config': config}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9000)