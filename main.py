import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Depends, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pytz

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ----------------- Load environment -----------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is not set in your .env file")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is not set in your .env file")

# ----------------- Database Setup ------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------- Car Marketplace Database Models ------------------
class CarListing(Base):
    __tablename__ = "car_listings"  # Correcting table name to match model
    id = Column(Integer, primary_key=True, index=True)
    make = Column(String, nullable=False)
    model = Column(String, nullable=False)
    year = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    description = Column(String, nullable=True)
    date_added = Column(DateTime, default=datetime.utcnow)

# ----------------- AI Agent Setup ------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory)

# ----------------- FastAPI App ------------------
app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
templates = Jinja2Templates(directory="templates")
bangladesh_tz = pytz.timezone('Asia/Dhaka')

# ----------------- Routes ------------------

# Home page to view recent car listings
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, db: Session = Depends(get_db)):
    # Fetching the most recent 10 car listings from the database
    car_listings = db.query(CarListing).order_by(CarListing.date_added.desc()).limit(10).all()
    for listing in car_listings:
        listing.date_added = listing.date_added.astimezone(bangladesh_tz)
    return templates.TemplateResponse("index.html", {"request": request, "car_listings": car_listings})

# Add new car listing
@app.post("/add_car_listing")
def add_car_listing(
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    price: float = Form(...),
    description: str = Form(None),
    db: Session = Depends(get_db),
):
    # Adding a new car listing to the database
    current_time_bst = datetime.now(bangladesh_tz)
    car_listing = CarListing(make=make, model=model, year=year, price=price, description=description, date_added=current_time_bst)
    db.add(car_listing)
    db.commit()

    # Calculate total car listings
    total_listings = db.query(CarListing).count()

    return JSONResponse({
        "message": f"✅ Car listing added: {make} {model} ({year}) for ${price}",
        "total_listings": total_listings
    })

# Clear all car listings
@app.post("/clear_car_listings")
def clear_car_listings(db: Session = Depends(get_db)):
    db.query(CarListing).delete()
    db.commit()
    return JSONResponse({"message": "✅ All car listings cleared."})

# ----------------- Chatbot Interaction (POST Endpoint) ------------------
@app.post("/chat")
async def chat_endpoint(
    request: Request,
    message: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    POST endpoint for chatbot interaction via form submission.
    """
    try:
        # Fetch latest car listings (limiting to 10 most recent)
        car_listings = db.query(CarListing).order_by(CarListing.date_added.desc()).limit(10).all()
        car_summary = "\n".join(
            [f"{cl.make} {cl.model} | {cl.year} | ${cl.price} | {cl.date_added.astimezone(bangladesh_tz).strftime('%Y-%m-%d %H:%M:%S')}" for cl in car_listings]
        ) or "No car listings available."

        # Build prompt with enhanced context
        prompt = f"""
        You are an AI assistant for a car marketplace company called "AutoHub". 
        Your role is to engage in friendly, professional, and informative conversation about the company, its services, and cars. 
        Use the following company information as context when responding:

        Company Overview:
        - Company Name: AutoHub
        - Industry: Online car marketplace
        - Founded: 2020
        - Mission: Simplify buying and selling cars with a trusted, seamless platform
        - Vision: Become the most reliable and customer-friendly car marketplace
        - Core Values: Trust, Transparency, Convenience, Customer Satisfaction, Innovation, Sustainability

        Services:
        1. Buying Cars
        2. Selling Cars
        3. Car Valuation & Market Insights
        4. Customer Support
        5. Community & Advice (tips on maintenance, buying guides, news)

        Recent Car Listings (latest 10):
        {car_summary}

        Tone & Personality:
        - Friendly, professional, approachable, trustworthy
        - Helpful and informative
        - Can provide casual conversation about cars or industry insights

        Instructions:
        - Answer the user question naturally and politely
        - Reference AutoHub services when relevant
        - Provide tips, insights, or guidance related to cars, buying, or selling
        - Avoid providing fake listings; only discuss marketplace features

        User asked: "{message}"
        """

        # Get AI response
        response = conversation.run(prompt)

        return JSONResponse({
            "success": True,
            "user_message": message,
            "ai_response": response,
            "car_listings_count": len(car_listings)
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Error processing chat request: {str(e)}"
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ----------------- Run ------------------
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)  # Create tables in the database
