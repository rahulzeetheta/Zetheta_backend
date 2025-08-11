from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Union

class UserSignup(BaseModel):
    email: Optional[EmailStr]
    phone: Optional[str]
    password: str
    first_name: str
    last_name: str
    country: str
    state: str

class UserLogin(BaseModel):
    email_or_phone: Union[EmailStr, str]
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class BehaviorInput(BaseModel):
    research_depth_value: int
    num_unique_sources: int
    decision_speed_category: str
    logins_per_month: int
    panic_sells_after_bad_news: bool
    buy_the_dip_after_downturn: bool
    strategic_rebalancing_after_news: bool
    transaction_reversals_per_month: int
    panic_selling_incidents: int
    financial_literacy_score: int
    volatility_level: int
    amount_invested: float

class TextPayload(BaseModel):
    text: str    

class QueryRequest(BaseModel):
    question: str