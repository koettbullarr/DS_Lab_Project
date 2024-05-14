from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pickle

app = FastAPI()

# Load the classifier and vectorizer
try:
    with open("./depl_model/classifier.pkl", "rb") as file:
        clf = pickle.load(file)
    with open("./depl_model/vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


# Define label mapping
label_mapping = {
    0: "interview",
    1: "letter",
    2: "comment"
}

# Serve static files from the 'docs' directory
app.mount("/static", StaticFiles(directory="docs"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("docs/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(text: str = Form(...)):
    try:
        input_vector = vectorizer.transform([text])
        prediction = clf.predict(input_vector)[0]  # Get the first (and only) prediction
        label = label_mapping.get(prediction, "Unknown Label")
        return JSONResponse(content={"label": label})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
