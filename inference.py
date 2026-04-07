from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/reset")
def reset():
    return {"status": "reset successful"}

if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000)