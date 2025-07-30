#!/usr/bin/env python3
"""
AI Children's Book Generator - Main Application
A FastAPI application that generates personalized children's books using AI.
"""

import os
import asyncio
import json
import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, validator
import uvicorn

# AI/ML imports
import openai
try:
    from diffusers import StableDiffusionPipeline
    import torch
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    print("⚠️  Diffusers not installed. Image generation will use placeholders.")

# Utility imports
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

class Config:
    """Application configuration"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Image generation settings
    IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "512"))
    IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "512"))
    
    # Safety settings
    ENABLE_SAFETY_FILTER = os.getenv("ENABLE_SAFETY_FILTER", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True

# Validate config on startup
Config.validate()

# =============================================================================
# DATA MODELS (Pydantic Schemas)
# =============================================================================

class BookRequest(BaseModel):
    """Request model for book generation"""
    child_name: str
    age: int
    interests: List[str]
    personality_traits: Optional[List[str]] = []
    book_length: Optional[str] = "short"  # short, medium, long
    illustration_style: Optional[str] = "cartoon"  # cartoon, watercolor, realistic
    
    @validator('child_name')
    def validate_child_name(cls, v):
        if not v or len(v.strip()) < 1:
            raise ValueError('Child name is required')
        if len(v) > 50:
            raise ValueError('Child name must be less than 50 characters')
        return v.strip()
    
    @validator('age')
    def validate_age(cls, v):
        if v < 3 or v > 12:
            raise ValueError('Age must be between 3 and 12 years')
        return v
    
    @validator('interests')
    def validate_interests(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one interest is required')
        if len(v) > 10:
            raise ValueError('Maximum 10 interests allowed')
        return [interest.strip() for interest in v if interest.strip()]

class StoryScene(BaseModel):
    """Individual scene in a story"""
    text: str
    illustration_prompt: str
    page_number: int

class GeneratedBook(BaseModel):
    """Response model for generated book"""
    book_id: str
    title: str
    scenes: List[StoryScene]
    illustrations: List[str]  # Base64 encoded images
    generation_time: float
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    ai_services: Dict[str, bool]

# =============================================================================
# AI SERVICES
# =============================================================================

class StoryGenerator:
    """Handles AI-powered story generation"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        logger.info("✅ StoryGenerator initialized")
    
    async def generate_story(self, request: BookRequest) -> Dict[str, Any]:
        """Generate a personalized story based on request parameters"""
        start_time = time.time()
        
        try:
            # Build personalized prompt
            prompt = self._build_prompt(request)
            
            # Generate story using OpenAI
            response = await asyncio.to_thread(
                self._call_openai_api,
                prompt
            )
            
            # Parse and validate response
            story_data = self._parse_story_response(response, request)
            
            generation_time = time.time() - start_time
            story_data['generation_time'] = generation_time
            
            logger.info(f"✅ Story generated in {generation_time:.2f}s for {request.child_name}")
            return story_data
            
        except Exception as e:
            logger.error(f"❌ Story generation failed: {str(e)}")
            # Return fallback story
            return self._create_fallback_story(request)
    
    def _build_prompt(self, request: BookRequest) -> str:
        """Build personalized prompt for story generation"""
        interests_str = ", ".join(request.interests)
        personality_str = ", ".join(request.personality_traits) if request.personality_traits else "curious and kind"
        
        length_guide = {
            "short": "4-5 short paragraphs",
            "medium": "6-8 paragraphs", 
            "long": "10-12 paragraphs"
        }
        
        return f"""
        Create a personalized children's story with these specifications:
        
        CHILD DETAILS:
        - Name: {request.child_name}
        - Age: {request.age} years old
        - Interests: {interests_str}
        - Personality: {personality_str}
        
        STORY REQUIREMENTS:
        - Length: {length_guide.get(request.book_length, '4-5 short paragraphs')}
        - Age-appropriate vocabulary for {request.age} years old
        - Include {request.child_name} as the main character
        - Incorporate their interests: {interests_str}
        - Include a positive moral lesson
        - Each paragraph should describe a scene that can be illustrated
        
        FORMAT your response as valid JSON:
        {{
            "title": "An engaging story title",
            "scenes": [
                {{
                    "text": "Scene paragraph text",
                    "illustration_prompt": "Description for illustration showing {request.child_name} and the scene",
                    "page_number": 1
                }},
                ...
            ]
        }}
        
        Ensure the story is engaging, educational, and perfectly suited for a {request.age}-year-old child.
        """
    
    def _call_openai_api(self, prompt: str) -> str:
        """Make synchronous call to OpenAI API"""
        response = self.client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE
        )
        return response.choices[0].message.content
    
    def _parse_story_response(self, response: str, request: BookRequest) -> Dict[str, Any]:
        """Parse and validate OpenAI response"""
        try:
            # Try to parse JSON response
            story_data = json.loads(response)
            
            # Validate required fields
            if 'title' not in story_data or 'scenes' not in story_data:
                raise ValueError("Invalid response format")
            
            # Ensure page numbers are set
            for i, scene in enumerate(story_data['scenes']):
                scene['page_number'] = i + 1
            
            return story_data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️  Failed to parse AI response: {e}")
            return self._create_fallback_story(request)
    
    def _create_fallback_story(self, request: BookRequest) -> Dict[str, Any]:
        """Create a fallback story when AI generation fails"""
        interests_str = ", ".join(request.interests[:2])  # Use first 2 interests
        
        return {
            "title": f"{request.child_name}'s Amazing Adventure",
            "scenes": [
                {
                    "text": f"Once upon a time, {request.child_name} discovered something magical related to {interests_str}.",
                    "illustration_prompt": f"{request.child_name} looking excited and curious",
                    "page_number": 1
                },
                {
                    "text": f"{request.child_name} learned that with curiosity and kindness, every day can be an adventure.",
                    "illustration_prompt": f"{request.child_name} smiling and happy with {interests_str}",
                    "page_number": 2
                }
            ],
            "generation_time": 0.1
        }

class IllustrationGenerator:
    """Handles AI-powered illustration generation"""
    
    def __init__(self):
        self.diffusion_available = DIFFUSION_AVAILABLE
        
        if self.diffusion_available:
            try:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                if torch.cuda.is_available():
                    self.pipe = self.pipe.to("cuda")
                    logger.info("✅ IllustrationGenerator initialized with GPU")
                else:
                    logger.info("✅ IllustrationGenerator initialized with CPU")
            except Exception as e:
                logger.warning(f"⚠️  Could not load Stable Diffusion: {e}")
                self.diffusion_available = False
        
        if not self.diffusion_available:
            logger.info("📝 IllustrationGenerator using placeholder mode")
    
    async def generate_illustration(self, prompt: str, child_name: str, style: str = "cartoon") -> str:
        """Generate illustration for a scene"""
        if self.diffusion_available:
            return await self._generate_ai_image(prompt, child_name, style)
        else:
            return self._create_placeholder_image(prompt, child_name)
    
    async def _generate_ai_image(self, prompt: str, child_name: str, style: str) -> str:
        """Generate image using Stable Diffusion"""
        try:
            full_prompt = f"{prompt}, featuring {child_name}, {style} style, children's book illustration, colorful, friendly, warm lighting"
            negative_prompt = "violence, scary, dark, inappropriate, adult content, weapons"
            
            # Generate image asynchronously
            image = await asyncio.to_thread(
                lambda: self.pipe(
                    full_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    height=Config.IMAGE_HEIGHT,
                    width=Config.IMAGE_WIDTH
                ).images[0]
            )
            
            # Convert to base64
            return self._image_to_base64(image)
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return self._create_placeholder_image(prompt, child_name)
    
    def _create_placeholder_image(self, prompt: str, child_name: str) -> str:
        """Create a colorful placeholder image"""
        img = Image.new('RGB', (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Add text to image
        text_lines = [
            f"Story Scene",
            f"Featuring {child_name}",
            prompt[:30] + "..." if len(prompt) > 30 else prompt
        ]
        
        y_position = Config.IMAGE_HEIGHT // 4
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x_position = (Config.IMAGE_WIDTH - text_width) // 2
            draw.text((x_position, y_position), line, fill='darkblue', font=font)
            y_position += 40
        
        return self._image_to_base64(img)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

class SafetyFilter:
    """Content safety and moderation"""
    
    def __init__(self):
        self.unsafe_keywords = {
            'violence': ['fight', 'hit', 'punch', 'kick', 'hurt', 'pain', 'blood', 'weapon'],
            'scary': ['scary', 'frightening', 'terror', 'nightmare', 'monster', 'ghost'],
            'inappropriate': ['adult', 'mature', 'inappropriate', 'unsuitable']
        }
        logger.info("✅ SafetyFilter initialized")
    
    def is_content_safe(self, text: str, child_age: int) -> tuple[bool, str]:
        """Check if content is safe for the specified age"""
        text_lower = text.lower()
        
        for category, keywords in self.unsafe_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return False, f"Contains {category} content: {keyword}"
        
        # Additional age-specific checks
        if child_age <= 5:
            complex_themes = ['death', 'divorce', 'complex emotions']
            for theme in complex_themes:
                if theme in text_lower:
                    return False, f"Too complex for age {child_age}: {theme}"
        
        return True, "Content is safe"
    
    def filter_story(self, story_data: Dict[str, Any], child_age: int) -> Dict[str, Any]:
        """Filter story content for safety"""
        filtered_scenes = []
        
        for scene in story_data.get('scenes', []):
            is_safe, reason = self.is_content_safe(scene['text'], child_age)
            if is_safe:
                filtered_scenes.append(scene)
            else:
                logger.warning(f"⚠️  Filtered scene: {reason}")
        
        # Ensure we have at least one scene
        if not filtered_scenes:
            filtered_scenes = [{
                "text": "And they all lived happily ever after!",
                "illustration_prompt": "A happy ending scene",
                "page_number": 1
            }]
        
        story_data['scenes'] = filtered_scenes
        return story_data

# =============================================================================
# APPLICATION SETUP
# =============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="AI Children's Book Generator",
    description="Generate personalized children's books using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI services
story_generator = StoryGenerator()
illustration_generator = IllustrationGenerator()
safety_filter = SafetyFilter()

# Global stats (in production, use a database)
app_stats = {
    "books_generated": 0,
    "total_generation_time": 0.0,
    "startup_time": datetime.now().isoformat()
}

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Children's Book Generator</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: bold; color: #555; }
            input, select, textarea { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; }
            input:focus, select:focus, textarea:focus { border-color: #667eea; outline: none; }
            button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; font-weight: bold; width: 100%; margin-top: 20px; }
            button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
            button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
            .result { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea; }
            .loading { display: none; text-align: center; color: #667eea; }
            .error { color: #dc3545; background: #f8d7da; padding: 15px; border-radius: 8px; margin-top: 20px; }
            .illustration { max-width: 200px; margin: 10px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 AI Children's Book Generator</h1>
            <p style="text-align: center; color: #666; margin-bottom: 30px;">Create personalized stories that spark imagination!</p>
            
            <form id="bookForm">
                <div class="form-group">
                    <label for="childName">Child's Name *</label>
                    <input type="text" id="childName" placeholder="Enter your child's name" required>
                </div>
                
                <div class="form-group">
                    <label for="age">Age *</label>
                    <select id="age" required>
                        <option value="">Select age</option>
                        <option value="3">3 years old</option>
                        <option value="4">4 years old</option>
                        <option value="5">5 years old</option>
                        <option value="6">6 years old</option>
                        <option value="7">7 years old</option>
                        <option value="8">8 years old</option>
                        <option value="9">9 years old</option>
                        <option value="10">10 years old</option>
                        <option value="11">11 years old</option>
                        <option value="12">12 years old</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="interests">Interests *</label>
                    <input type="text" id="interests" placeholder="e.g., dinosaurs, space, magic, animals" required>
                    <small style="color: #666;">Separate multiple interests with commas</small>
                </div>
                
                <div class="form-group">
                    <label for="personality">Personality Traits</label>
                    <input type="text" id="personality" placeholder="e.g., curious, brave, kind, funny">
                    <small style="color: #666;">Optional: helps personalize the story</small>
                </div>
                
                <div class="form-group">
                    <label for="bookLength">Story Length</label>
                    <select id="bookLength">
                        <option value="short">Short (4-5 scenes)</option>
                        <option value="medium">Medium (6-8 scenes)</option>
                        <option value="long">Long (10-12 scenes)</option>
                    </select>
                </div>
                
                <button type="submit" id="generateBtn">✨ Generate Magical Story ✨</button>
            </form>
            
            <div class="loading" id="loading">
                <h3>🎨 Creating your magical story...</h3>
                <p>This may take 1-2 minutes while our AI crafts the perfect adventure!</p>
                <div style="margin: 20px 0;">
                    <div style="width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto;"></div>
                </div>
            </div>
            
            <div class="result" id="result" style="display: none;">
                <h2 id="bookTitle"></h2>
                <div id="bookContent"></div>
                <div id="bookIllustrations"></div>
                <div id="bookStats" style="margin-top: 20px; font-size: 14px; color: #666;"></div>
            </div>
        </div>

        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>

        <script>
            document.getElementById('bookForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = {
                    child_name: document.getElementById('childName').value.trim(),
                    age: parseInt(document.getElementById('age').value),
                    interests: document.getElementById('interests').value.split(',').map(s => s.trim()).filter(s => s),
                    personality_traits: document.getElementById('personality').value ? 
                        document.getElementById('personality').value.split(',').map(s => s.trim()).filter(s => s) : [],
                    book_length: document.getElementById('bookLength').value
                };
                
                // Validation
                if (!formData.child_name || !formData.age || formData.interests.length === 0) {
                    alert('Please fill in all required fields!');
                    return;
                }
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                document.getElementById('generateBtn').disabled = true;
                
                try {
                    const response = await fetch('/generate-book', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(formData)
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Generation failed');
                    }
                    
                    const book = await response.json();
                    
                    // Display results
                    document.getElementById('bookTitle').textContent = book.title;
                    
                    let contentHtml = '<h3>📖 Your Story:</h3>';
                    book.scenes.forEach((scene, index) => {
                        contentHtml += `<div style="margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px;">
                            <h4>Page ${scene.page_number}</h4>
                            <p style="line-height: 1.6;">${scene.text}</p>
                        </div>`;
                    });
                    document.getElementById('bookContent').innerHTML = contentHtml;
                    
                    // Display illustrations
                    if (book.illustrations && book.illustrations.length > 0) {
                        let illustrationsHtml = '<h3>🎨 Illustrations:</h3><div style="text-align: center;">';
                        book.illustrations.forEach((img, index) => {
                            illustrationsHtml += `<img src="data:image/png;base64,${img}" class="illustration" alt="Illustration ${index + 1}">`;
                        });
                        illustrationsHtml += '</div>';
                        document.getElementById('bookIllustrations').innerHTML = illustrationsHtml;
                    }
                    
                    // Display stats
                    document.getElementById('bookStats').innerHTML = `
                        <strong>Generation Details:</strong><br>
                        • Generated in ${book.generation_time.toFixed(2)} seconds<br>
                        • ${book.scenes.length} scenes created<br>
                        • Book ID: ${book.book_id}
                    `;
                    
                    document.getElementById('result').style.display = 'block';
                    
                } catch (error) {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'error';
                    errorDiv.innerHTML = `<strong>Oops!</strong> ${error.message}`;
                    document.getElementById('result').appendChild(errorDiv);
                    document.getElementById('result').style.display = 'block';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('generateBtn').disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        ai_services={
            "openai": bool(Config.OPENAI_API_KEY),
            "stable_diffusion": illustration_generator.diffusion_available,
            "safety_filter": True
        }
    )

@app.post("/generate-book", response_model=GeneratedBook)
async def generate_book(request: BookRequest, background_tasks: BackgroundTasks):
    """Generate a personalized children's book"""
    start_time = time.time()
    
    try:
        # Generate unique book ID
        book_id = f"book_{int(time.time() * 1000)}"
        
        # Generate story
        logger.info(f"🎨 Generating story for {request.child_name}, age {request.age}")
        story_data = await story_generator.generate_story(request)
        
        # Apply safety filter
        if Config.ENABLE_SAFETY_FILTER:
            story_data = safety_filter.filter_story(story_data, request.age)
        
        # Generate illustrations
        illustrations = []
        if story_data.get('scenes'):
            for scene in story_data['scenes']:
                illustration = await illustration_generator.generate_illustration(
                    scene['illustration_prompt'],
                    request.child_name,
                    request.illustration_style
                )
                illustrations.append(illustration)
        
        # Create response
        total_time = time.time() - start_time
        
        response = GeneratedBook(
            book_id=book_id,
            title=story_data['title'],
            scenes=[StoryScene(**scene) for scene in story_data['scenes']],
            illustrations=illustrations,
            generation_time=total_time,
            metadata={
                "child_age": request.age,
                "interests": request.interests,
                "book_length": request.book_length,
                "illustration_style": request.illustration_style,
                "scenes_count": len(story_data['scenes']),
                "illustrations_count": len(illustrations)
            }
        )
        
        # Update stats in background
        background_tasks.add_task(update_stats, total_time)
        
        logger.info(f"✅ Book generated successfully in {total_time:.2f}s - ID: {book_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Book generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Book generation failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get application statistics"""
    return {
        "books_generated": app_stats["books_generated"],
        "average_generation_time": (
            app_stats["total_generation_time"] / app_stats["books_generated"] 
            if app_stats["books_generated"] > 0 else 0
        ),
        "uptime_hours": (
            datetime.now() - datetime.fromisoformat(app_stats["startup_time"])
        ).total_seconds() / 3600,
        "ai_services": {
            "openai_available": bool(Config.OPENAI_API_KEY),
            "diffusion_available": illustration_generator.diffusion_available,
            "gpu_available": torch.cuda.is_available() if DIFFUSION_AVAILABLE else False
        }
    }

# =============================================================================
# BACKGROUND TASKS & UTILITIES
# =============================================================================

async def update_stats(generation_time: float):
    """Update application statistics"""
    app_stats["books_generated"] += 1
    app_stats["total_generation_time"] += generation_time

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🚀 Starting AI Children's Book Generator")
    logger.info(f"📝 OpenAI API: {'✅ Connected' if Config.OPENAI_API_KEY else '❌ Missing'}")
    logger.info(f"🎨 Image Generation: {'✅ Available' if illustration_generator.diffusion_available else '📝 Placeholder mode'}")
    logger.info(f"🛡️  Safety Filter: {'✅ Enabled' if Config.ENABLE_SAFETY_FILTER else '⚠️  Disabled'}")
    logger.info("🌟 Ready to generate magical stories!")

if __name__ == "__main__":
    print("=" * 60)
    print("🎨 AI Children's Book Generator")
    print("=" * 60)
    print("📋 Setup Checklist:")
    print(f"   • OpenAI API Key: {'✅' if Config.OPENAI_API_KEY else '❌ Set OPENAI_API_KEY environment variable'}")
    print(f"   • GPU Available: {'✅' if torch.cuda.is_available() and DIFFUSION_AVAILABLE else '⚠️  CPU mode (slower image generation)'}")
    print(f"   • Dependencies: {'✅' if DIFFUSION_AVAILABLE else '⚠️  Run: pip install torch diffusers'}")
    print("\n🚀 Starting server...")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Web Interface: http://localhost:8000")
    print("   • Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
