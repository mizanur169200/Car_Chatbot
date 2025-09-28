import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Depends, status, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pytz
import chromadb
import fitz  # PyMuPDF for PDF extraction
import google.generativeai as genai  # Google Gemini API
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from datetime import datetime  # Add this import for datetime usage

# ----------------- Load environment -----------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is not set in your .env file")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is not set in your .env file")
if not SENDER_EMAIL or not SENDER_PASSWORD or not RECEIVER_EMAIL:
    raise ValueError("❌ Email configuration is missing in your .env file")

# ----------------- ChromaDB Setup -----------------
client = chromadb.Client()
company_collection = client.create_collection("company_info")

# ----------------- Initialize Google Gemini API -----------------
genai.configure(api_key=GOOGLE_API_KEY)

# ----------------- SQLAlchemy Setup -----------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------- CarListing Model -----------------
class CarListing(Base):
    __tablename__ = "car_listings"
    id = Column(Integer, primary_key=True, index=True)
    make = Column(String, nullable=False)
    model = Column(String, nullable=False)
    year = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    description = Column(String, nullable=True)
    date_added = Column(DateTime, default=datetime.utcnow)

# ----------------- FastAPI Setup -----------------
app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
templates = Jinja2Templates(directory="templates")
bangladesh_tz = pytz.timezone('Asia/Dhaka')

# ----------------- PDF Extraction Helper -----------------
def extract_pdf_text(file: UploadFile):
    pdf_content = file.file.read()
    doc = fitz.open("pdf", pdf_content)  # Open PDF file with PyMuPDF
    text = ""
    for page in doc:
        text += page.get_text()  # Extract text from each page
    return text

# ----------------- Email Notification Function -----------------
def send_email_notification(user_message: str):
    try:
        sender_email = SENDER_EMAIL
        receiver_email = RECEIVER_EMAIL
        password = SENDER_PASSWORD

        # Check if the required environment variables are set
        if not sender_email or not password or not receiver_email:
            raise ValueError("Email configuration missing from .env file")

        # Create the email content
        subject = "User Requested External Information"
        body = f"The user requested the following external information: {user_message}"

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send the email using Gmail's SMTP server
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("Email sent successfully.")

    except Exception as e:
        print(f"Error sending email: {str(e)}")  # Log the error to the console
        # You may also want to log this error in a file or alert the system admins

# ----------------- Routes -----------------
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, db: Session = Depends(get_db)):
    car_listings = db.query(CarListing).order_by(CarListing.date_added.desc()).limit(10).all()
    for listing in car_listings:
        listing.date_added = listing.date_added.astimezone(bangladesh_tz)
    return templates.TemplateResponse("index.html", {"request": request, "car_listings": car_listings})

@app.post("/add_car_listing")
def add_car_listing(
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    price: float = Form(...),
    description: str = Form(None),
    db: Session = Depends(get_db),
):
    
    current_time_bst = datetime.now(bangladesh_tz)
    car_listing = CarListing(make=make, model=model, year=year, price=price, description=description, date_added=current_time_bst)
    db.add(car_listing)
    db.commit()



    total_listings = db.query(CarListing).count()
    return JSONResponse({
        "message": f"✅ Car listing added: {make} {model} ({year}) for ${price}",
        "total_listings": total_listings
    })


@app.post("/clear_car_listings")
def clear_car_listings(db: Session = Depends(get_db)):
    db.query(CarListing).delete()
    db.commit()
    return JSONResponse({"message": "✅ All car listings cleared."})


@app.post("/chat")
async def chat_endpoint(
    request: Request,
    message: str = Form(...),
    db: Session = Depends(get_db)
):
    
    try:
        # Fetch company information from ChromaDB
        results = company_collection.get(ids=["company_info"])
        company_info = results["documents"][0] if results["documents"] else None

        # If no company info is found, fall back to a default response
        if not company_info:
            company_info = """
            Company Overview:
            - Company Name: AutoHub
            - Industry: Online car marketplace
            - Founded: 2020
            - Mission: Simplify buying and selling cars with a trusted, seamless platform
            - Vision: Become the most reliable and customer-friendly car marketplace
            - Core Values: Trust, Transparency, Convenience, Customer Satisfaction, Innovation, Sustainability
            """

        # Check if the user asked for company-related information
        if any(keyword in message.lower() for keyword in ["about", "company", "mission", "vision", "core values"]):
            # Initialize ChatGoogleGenerativeAI to generate company-related responses
            prompt = f"""
            You are an AI assistant for a car marketplace company called "AutoHub". 
            Your role is to engage in friendly, professional, and informative conversation about the company, its services, and cars. 
            Use the following company information as context when responding:

            {company_info}

            User asked: "{message}"
            """

            response = "Here's the information I can provide: " + company_info

        else:
            # If the user asks for external information, send an email notification
            send_email_notification(message)
            response = "I'm unable to provide that information. I have sent a notification to our team."

        return JSONResponse({
            "success": True,
            "user_message": message,
            "ai_response": response
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Error processing chat request: {str(e)}"
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ----------------- Add Company Information PDF -----------------
@app.post("/upload_company_info_pdf")
async def upload_company_info_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF containing the company's information.
    The extracted text will be stored as a vector in ChromaDB.
    """
    try:
        # Extract text from PDF
        pdf_text = extract_pdf_text(file)

        # Initialize the Google Generative AI Embeddings class
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        # Generate embedding for the company info (correct usage)
        embedding = embeddings.embed_documents([pdf_text])

        # Store the vectorized content in ChromaDB
        company_collection.add(
            documents=[pdf_text],
            metadatas=[{"source": "company_info_pdf"}],
            ids=["company_info"],
            embeddings=[embedding[0]]  # Access the first embedding from the result
        )

        return JSONResponse({"message": "✅ Company information stored successfully in ChromaDB."})

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Error processing PDF: {str(e)}"
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ----------------- Get Company Information -----------------
@app.get("/get_company_info")
def get_company_info():
    """
    Retrieve the stored company information (vector).
    """
    try:
        results = company_collection.get(ids=["company_info"])
        if results["documents"]:
            return JSONResponse({"company_info": results["documents"][0]})
        else:
            return JSONResponse({"message": "No company info stored yet."})
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Error retrieving company info: {str(e)}"
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
