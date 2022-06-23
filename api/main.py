from fastapi import FastAPI
from typing import Dict
from datatypes import *
from utils.interpolation import linear_interpolation
import pandas as pd

app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'Hello World!'}


@app.post('/interpolation/simple')
async def interpol(data: Dict, config: Config):
    df = pd.DataFrame.from_dict(data)
    data = None
    if config.interpolation == InterMethod.linear:
        data = linear_interpolation(df, config)

    return {'data': data}



@app.post('/interpolation/timeswitch')
async def tinterpol(data: Dict[str, float], config: ConfigTimeSwitch):
    return {'data': data, 'config': config}


@app.post('/detect/outtier')
async def detection(data: Dict[str, float], config: ConfigDetection):
    return {'data': data, 'config': config}


@app.post('/management/balanced')
async def balanced(data: Dict[str, float], config: ConfigBalanced):
    return {'data': data, 'config': config}
