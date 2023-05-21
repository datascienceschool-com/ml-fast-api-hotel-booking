from fastapi import FastAPI
from app.routes import predict

# Create instance of FastAPI
api = FastAPI()

# Add our custom routes to handle traffic 
@api.get('/', tags=['greetings'])
async def greet_user():
    return {'message': 'Hello and welcome to the Data Science School ML API!'}

# Predict route which will serve predictions from our custom models
api.include_router(predict.predict_router, tags=['predict'])
