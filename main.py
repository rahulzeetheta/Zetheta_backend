from fastapi import FastAPI, HTTPException, Depends
from db import users_collection
from models import UserSignup, UserLogin, TokenResponse,QueryRequest
from auth.auth_handler import hash_password, verify_password, create_access_token
from auth.auth_bearer import JWTBearer
from bson.objectid import ObjectId
from typing import Union
from behavior_model import get_behavioral_score
from models import BehaviorInput,TextPayload
from fastapi import FastAPI, Query
from model import fetch_data, preprocess, load_lstm_model, predict_next
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, HTTPException
from ai_models.finbert_model import get_finbert_sentiment
from chatbot.bot import get_response
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.post("/signup", response_model=TokenResponse)
async def signup(user: UserSignup):
    # Check if email or phone already exists
    existing_user = await users_collection.find_one({
        "$or": [{"email": user.email}, {"phone": user.phone}]
    })
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    user_dict = user.dict()
    user_dict["password"] = hash_password(user.password)

    res = await users_collection.insert_one(user_dict)
    token = create_access_token({"user_id": str(res.inserted_id)})
    return {"access_token": token}

@app.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    query = {"$or": [{"email": user.email_or_phone}, {"phone": user.email_or_phone}]}
    db_user = await users_collection.find_one(query)

    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"user_id": str(db_user["_id"])})
    return {"access_token": token}

@app.get("/protected", dependencies=[Depends(JWTBearer())])
async def protected():
    return {"message": "Access granted to protected route"}

@app.post("/analyze-behavior", dependencies=[Depends(JWTBearer())])
async def analyze_behavior(data: BehaviorInput):
    score = get_behavioral_score(**data.dict())
    return {"behavioral_risk_score": score}
model = load_lstm_model()


#chatbots
chat_history = []


@app.post("/chatbot")
def chat_with_bot(req: QueryRequest):
    global chat_history
    answer = get_response(req.question, chat_history)
    chat_history.append((req.question, answer))
    return {"answer": answer}