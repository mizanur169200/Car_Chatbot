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

# ----------------- Database Query Functions -----------------
def get_car_listings_from_db(db: Session, filters: dict = None):
    """Get car listings from database with optional filters"""
    query = db.query(CarListing)
    
    if filters:
        if 'make' in filters:
            query = query.filter(CarListing.make.ilike(f"%{filters['make']}%"))
        if 'model' in filters:
            query = query.filter(CarListing.model.ilike(f"%{filters['model']}%"))
        if 'min_year' in filters:
            query = query.filter(CarListing.year >= filters['min_year'])
        if 'max_year' in filters:
            query = query.filter(CarListing.year <= filters['max_year'])
        if 'min_price' in filters:
            query = query.filter(CarListing.price >= filters['min_price'])
        if 'max_price' in filters:
            query = query.filter(CarListing.price <= filters['max_price'])
    
    return query.order_by(CarListing.date_added.desc()).all()

def format_car_listings_for_ai(car_listings):
    """Format car listings data for AI response"""
    if not car_listings:
        return "No car listings found in the database."
    
    formatted = "Current Car Listings in Database:\n"
    for i, car in enumerate(car_listings, 1):
        formatted += f"{i}. {car.year} {car.make} {car.model} - ${car.price}"
        if car.description:
            formatted += f" - {car.description}"
        formatted += f" (Added: {car.date_added.strftime('%Y-%m-%d')})\n"
    
    return formatted

def extract_car_filters_from_message(message: str):
    """Extract car search filters from user message"""
    filters = {}
    message_lower = message.lower()
    
    # Extract make
    makes = ['toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 'nissan', 'hyundai', 'kia', 'chevrolet']
    for make in makes:
        if make in message_lower:
            filters['make'] = make
            break
    
    # Extract model keywords
    models = ['camry', 'civic', 'corolla', 'accord', 'mustang', 'x5', 'a4', 'sentra', 'elantra', 'sportage']
    for model in models:
        if model in message_lower:
            filters['model'] = model
            break
    
    # Extract year ranges
    words = message_lower.split()
    for i, word in enumerate(words):
        if word.isdigit() and len(word) == 4 and 1900 <= int(word) <= 2030:
            filters['min_year'] = int(word)
            # Check if there's a range like "2015 to 2020"
            if i + 2 < len(words) and words[i+1] in ['to', 'until', 'up'] and words[i+2].isdigit():
                filters['max_year'] = int(words[i+2])
            break
    
    # Extract price ranges
    if 'under' in message_lower or 'less than' in message_lower:
        for word in words:
            if word.replace('$', '').replace(',', '').isdigit():
                filters['max_price'] = float(word.replace('$', '').replace(',', ''))
                break
    elif 'over' in message_lower or 'more than' in message_lower:
        for word in words:
            if word.replace('$', '').replace(',', '').isdigit():
                filters['min_price'] = float(word.replace('$', '').replace(',', ''))
                break
    elif 'between' in message_lower:
        prices = []
        for word in words:
            clean_word = word.replace('$', '').replace(',', '')
            if clean_word.isdigit():
                prices.append(float(clean_word))
                if len(prices) == 2:
                    filters['min_price'] = min(prices)
                    filters['max_price'] = max(prices)
                    break
    
    return filters

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

        # If no company info is found, create a default
        if not company_info:
            company_info = """
            Company Overview:
            - Company Name: AutoHub
            - Industry: Online car marketplace
            - Founded: 2020
            - Mission: Simplify buying and selling cars with a trusted, seamless platform
            - Vision: Become the most reliable and customer-friendly car marketplace
            - Core Values: Trust, Transparency, Convenience, Customer Satisfaction, Innovation, Sustainability
            - Services: Car listings, vehicle information, marketplace services
            """

        # Initialize Gemini model for conversational AI
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Check if the message is about car listings in the database
        user_message_lower = message.lower()
        car_listings_data = ""
        
        # Check if user is asking about cars in the database
        car_db_keywords = ['car listings', 'available cars', 'cars in database', 'show me cars', 'list cars', 
                          'what cars do you have', 'inventory', 'vehicles available', 'car database']
        
        is_car_database_query = any(keyword in user_message_lower for keyword in car_db_keywords)
        
        if is_car_database_query:
            # Extract filters from user message
            filters = extract_car_filters_from_message(message)
            # Get car listings from database
            car_listings = get_car_listings_from_db(db, filters)
            car_listings_data = format_car_listings_for_ai(car_listings)
        
        # Define the system prompt with expanded capabilities including database queries
        system_prompt = f"""
        You are an AI assistant for AutoHub, a car marketplace company. Your responses must follow these rules:

        1. EXPANDED GENERAL CONVERSATION: Engage naturally in versatile casual conversations including:
        
           GREETINGS & FAREWELLS:
           - "hi", "hello", "hey", "good morning/afternoon/evening/night"
           - "howdy", "greetings", "what's happening", "yo"
           - "goodbye", "bye", "see you", "take care", "farewell", "catch you later"
           
           PERSONAL WELL-BEING:
           - "how are you", "how's it going", "what's up", "how do you do"
           - "how have you been", "how's your day", "how's everything"
           - Respond positively but briefly, then redirect to offering help
           
           SMALL TALK & CASUAL TOPICS:
           - Weather: "nice weather", "rainy day", "hot/cold today" 
           - Time: "what time is it" (respond generically without real-time)
           - Day: "what day is it", "happy monday/friday/weekend"
           - Compliments: "you're helpful", "thanks for helping", "good job"
           - Emotions: "I'm happy/sad/bored/tired" (show empathy)
           - Hobbies: "what do you like to do" (talk about helping users)
           
           INTRODUCTIONS & IDENTITY:
           - "who are you", "what's your name", "tell me about yourself"
           - "what can you do", "what are your capabilities"
           - Explain you're an AI assistant for AutoHub
           
           APPRECIATION & FEEDBACK:
           - "thank you", "thanks", "appreciate it", "you're awesome"
           - "I like you", "you're helpful", "good bot"
           - Respond graciously and offer further assistance
           
           CONTINUING CONVERSATION:
           - Follow-up questions: "and you?", "what about you?"
           - Conversation continuers: "that's interesting", "tell me more"
           - Clarifying questions: "what do you mean?", "can you explain?"

        2. COMPANY INFORMATION: Provide information ONLY based on this company data:
        {company_info}
        
        3. CAR-RELATED QUERIES: You can discuss general car topics, vehicles, automotive concepts, buying/selling processes
        
        4. DATABASE QUERIES: When the user asks about car listings, available cars, or database information, 
           use this car listings data from our database to provide accurate information:
           {car_listings_data if car_listings_data else "No specific car listings data available for this query."}
        
        5. EXTERNAL KNOWLEDGE RESTRICTION: For any question outside general conversation, company/car information, 
           or database queries, respond:
           "I am unable to provide that information as it falls outside my knowledge scope. I can only help with general conversation, AutoHub company information, and our car listings."

        Conversation Style:
        - Be friendly, warm, and engaging in general conversation
        - Keep responses conversational but not overly lengthy
        - Show personality while maintaining professionalism
        - When providing database information, be accurate and helpful
        - When in doubt about topic boundaries, err on the side of caution

        Examples:
        User: "hey there! how's your day going?" → "Hello! I'm functioning well today, thanks for asking! How can I help you with AutoHub?"
        User: "what's your mission?" → Explain AutoHub's mission from company info
        User: "show me available cars" → Provide information from the car listings database
        User: "do you have any Toyota cars under $20000?" → Check database and provide filtered results
        User: "what's the weather?" → External knowledge restriction response

        Current user message: "{message}"
        """

        # Generate response using Gemini
        response = model.generate_content(system_prompt)
        
        # Check if the message falls within allowed categories
        user_message_lower = message.lower()
        
        # Expanded general conversation triggers
        greeting_words = ['hi', 'hello', 'hey', 'howdy', 'greetings', 'yo', 'good morning', 'good afternoon', 'good evening', 'good night']
        wellbeing_words = ['how are you', 'how you', 'how\'s it going', 'what\'s up', 'how do you do', 'how have you been', 'how\'s your day', 'how\'s everything']
        farewell_words = ['bye', 'goodbye', 'see you', 'take care', 'farewell', 'later', 'cya']
        smalltalk_words = ['weather', 'day', 'time', 'today', 'tonight', 'weekend', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        appreciation_words = ['thank', 'thanks', 'appreciate', 'awesome', 'good job', 'well done', 'helpful', 'like you']
        identity_words = ['who are you', 'what are you', 'your name', 'tell me about yourself', 'what can you do', 'your capabilities']
        emotional_words = ['happy', 'sad', 'bored', 'tired', 'excited', 'angry', 'frustrated', 'annoyed', 'great', 'awesome', 'wonderful', 'terrible']
        continuation_words = ['and you', 'what about you', 'that\'s interesting', 'tell me more', 'explain', 'what do you mean']
        
        is_general_conversation = any(
            any(word in user_message_lower for word in word_list) 
            for word_list in [greeting_words, wellbeing_words, farewell_words, smalltalk_words, 
                            appreciation_words, identity_words, emotional_words, continuation_words]
        )
        
        is_company_related = any(word in user_message_lower for word in ['about', 'company', 'mission', 'vision', 'core values', 'autohub', 'services', 'founded', 'overview'])
        is_car_related = any(word in user_message_lower for word in ['car', 'vehicle', 'auto', 'marketplace', 'buy', 'sell', 'listings', 'price', 'model', 'make', 'year', 'mileage', 'dealer'])
        is_database_query = any(keyword in user_message_lower for keyword in car_db_keywords) or is_car_database_query

        # If it's neither general conversation, company-related, car-related, nor database query, it's external knowledge
        if not (is_general_conversation or is_company_related or is_car_related or is_database_query):
            # Send email notification with the actual user message
            send_email_notification(message)
            response_text = "I am unable to provide that information as it falls outside my knowledge scope. I can only help with general conversation, AutoHub company information, and our car listings."
        else:
            response_text = response.text

        return JSONResponse({
            "success": True,
            "user_message": message,
            "ai_response": response_text
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

# ----------------- Get Car Listings API -----------------
@app.get("/car_listings")
def get_car_listings_api(
    make: str = None,
    model: str = None,
    min_year: int = None,
    max_year: int = None,
    min_price: float = None,
    max_price: float = None,
    db: Session = Depends(get_db)
):
    """
    API endpoint to get car listings with filters
    """
    filters = {}
    if make: filters['make'] = make
    if model: filters['model'] = model
    if min_year: filters['min_year'] = min_year
    if max_year: filters['max_year'] = max_year
    if min_price: filters['min_price'] = min_price
    if max_price: filters['max_price'] = max_price
    
    car_listings = get_car_listings_from_db(db, filters)
    
    # Convert to list of dictionaries for JSON response
    listings_data = []
    for car in car_listings:
        listings_data.append({
            "id": car.id,
            "make": car.make,
            "model": car.model,
            "year": car.year,
            "price": car.price,
            "description": car.description,
            "date_added": car.date_added.isoformat()
        })
    
    return JSONResponse({
        "success": True,
        "car_listings": listings_data,
        "total_count": len(listings_data)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)