from fastapi import FastAPI

# Define App
app = FastAPI()

# Routes
@app.get("/healthz")
async def health_check() -> dict[str, str]:
    """Check status of Web-Server"""
    return {"status": "ok"}

@app.post("/greet")
async def greet_user(names: list[str]) -> dict[str,str]:
    """Greet the user when given a list of users"""
    output_messege: dict[str, str] = {}
    for name in names:
        output_messege[name] = f"Hello User: {name}"
    
    return output_messege