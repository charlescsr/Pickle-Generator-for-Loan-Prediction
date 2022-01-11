from pathlib import Path
import pickle
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import aiofiles
import pandas as pd
from flaml import AutoML

app = FastAPI()

# String of current working directory
location = '/home/csr/Documents/dbapi/'
automl = AutoML()
automl_settings = {
    "time_budget": 10,
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": './train_log.log',
}

@app.post("/train/", response_class=FileResponse)
async def train(data: UploadFile = File(...)):
    if os.path.exists("model.pickle"):
        os.remove("model.pickle")

    # Remove train log file
    if os.path.exists("train_log.log"):
        os.remove("train_log.log")

    async with aiofiles.open(location+data.filename, 'wb') as dataset:
        content = await data.read()
        await dataset.write(content)
    await dataset.close()

    # Load the dataset
    df = pd.read_csv(location+data.filename)

    # Train the dataset
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    automl.fit(X, y, **automl_settings)

    # Save the model
    with open("model.pickle", "wb") as file:
        pickle.dump(automl, file, pickle.HIGHEST_PROTOCOL)

    file.close()

    # Remove the dataset
    os.remove(location+data.filename)

    return FileResponse(Path('model.pickle'), filename='model.pickle', media_type='.pickle')

if __name__ == "__main__":
    uvicorn.run("main:app")