# main.py - Fixed Version - Error-Free Backend
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional, Union, Any
import json
import asyncio
from datetime import datetime, timedelta
import uuid
import os
import re
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="IRC-Style Chat with ChatAI", version="2.0.1")

# Environment variables
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
MAX_MESSAGES_PER_CHANNEL = int(os.getenv("MAX_MESSAGES_PER_CHANNEL", "1000"))
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))

# CORS middleware with security improvements
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# =============================================================================
# DATA MODELS WITH VALIDATION
# =============================================================================

class User(BaseModel):
    user_id: str
    email: EmailStr  # Better email validation
    display_name: str
    avatar_url: Optional[str] = None
    status: str = "online"
    last_seen: datetime
    nick_name: Optional[str] = None

class Channel(BaseModel):
    channel_id: str
    channel_name: str
    description: Optional[str] = None
    last_message: Optional[str] = None
    last_message_time: Optional[datetime] = None
    member_count: int = 0
    created_by: str
    is_private: bool = False

class Message(BaseModel):
    message_id: str
    channel_id: str
    user_email: EmailStr
    display_name: str
    message_text: str
    timestamp: datetime
    is_ai: bool = False
    message_type: str = "message"
    command_used: Optional[str] = None

class SendMessageRequest(BaseModel):
    channel_id: str
    user_email: EmailStr
    display_name: str
    message_text: str

    # Validation
    class Config:
        str_strip_whitespace = True

class JoinChannelRequest(BaseModel):
    channel_id: str
    user_email: EmailStr
    display_name: str

# =============================================================================
# IMPROVED IN-MEMORY STORAGE WITH CLEANUP
# =============================================================================

users_db: Dict[str, User] = {}
channels_db: Dict[str, Channel] = {}
messages_db: List[Message] = []
channel_members: Dict[str, List[str]] = {}
ai_context: Dict[str, List[Message]] = {}

# Memory management
def cleanup_old_messages():
    """Remove old messages to prevent memory issues"""
    global messages_db
    if len(messages_db) > MAX_MESSAGES_PER_CHANNEL * 3:  # Keep 3x max per channel
        # Sort by timestamp and keep recent ones
        messages_db.sort(key=lambda x: x.timestamp, reverse=True)
        messages_db = messages_db[:MAX_MESSAGES_PER_CHANNEL * 3]
        logger.info(f"Cleaned up old messages. Current count: {len(messages_db)}")

# =============================================================================
# IMPROVED CHATAI PERSONALITY SYSTEM
# =============================================================================

class ChatAIPersonality:
    def __init__(self):
        self.channel_vibes = {}
        self.user_interactions = {}  # Track user interaction patterns
        
    def detect_vibe(self, recent_messages: List[Message]) -> str:
        """Enhanced vibe detection with better logic"""
        if not recent_messages:
            return "neutral"
            
        # Analyze last 5 messages
        recent_texts = [msg.message_text.lower() for msg in recent_messages[-5:]]
        combined_text = " ".join(recent_texts)
        
        # Enhanced indicators
        fun_indicators = ["haha", "lol", "😂", "😄", "😊", "cool", "awesome", "nice", "funny", "joke"]
        serious_indicators = ["help", "problem", "advice", "confused", "difficult", "important", "urgent", "?"]
        sad_indicators = ["sad", "bad", "terrible", "awful", "down", "😢", "😞", "depressed", "upset"]
        excited_indicators = ["amazing", "excited", "wow", "great", "fantastic", "love", "❤️", "🎉"]
        
        fun_score = sum(1 for indicator in fun_indicators if indicator in combined_text)
        serious_score = sum(1 for indicator in serious_indicators if indicator in combined_text)
        sad_score = sum(1 for indicator in sad_indicators if indicator in combined_text)
        excited_score = sum(1 for indicator in excited_indicators if indicator in combined_text)
        
        # Decision logic
        if sad_score > 0:
            return "supportive"
        elif excited_score > 1:
            return "enthusiastic"
        elif fun_score > serious_score:
            return "chill"
        elif serious_score > 0:
            return "socratic"
        else:
            return "neutral"
    
    def get_response_style(self, vibe: str) -> Dict[str, str]:
        """Enhanced response styles"""
        styles = {
            "chill": {
                "emoji": "😎",
                "tone": "relaxed and humorous",
                "approach": "casual with light wisdom"
            },
            "socratic": {
                "emoji": "🤔", 
                "tone": "thoughtful and questioning",
                "approach": "provocative questions with insight"
            },
            "supportive": {
                "emoji": "💙",
                "tone": "empathetic and encouraging", 
                "approach": "supportive with gentle guidance"
            },
            "enthusiastic": {
                "emoji": "🚀",
                "tone": "energetic and motivational",
                "approach": "matching excitement with wisdom"
            },
            "neutral": {
                "emoji": "🤖",
                "tone": "balanced and helpful",
                "approach": "informative with personality"
            }
        }
        return styles.get(vibe, styles["neutral"])

chatai_personality = ChatAIPersonality()

# =============================================================================
# ENHANCED CHATAI RESPONSE GENERATOR
# =============================================================================

async def generate_chatai_response(channel_id: str, trigger_message: str, context: List[Message]) -> str:
    """Enhanced ChatAI with better responses and error handling"""
    try:
        # Detect conversation vibe
        vibe = chatai_personality.detect_vibe(context)
        style = chatai_personality.get_response_style(vibe)
        
        # Store vibe for this channel
        chatai_personality.channel_vibes[channel_id] = vibe
        
        # Enhanced response generation based on vibe and content
        if vibe == "chill":
            responses = [
                f"{style['emoji']} Yo! Văd că e vibe bun pe aici! Dar de ce mă întrebi pe mine când poți să experimentezi singur? 😄",
                f"{style['emoji']} Haha, good vibes! Dar înainte să îți răspund... tu ce crezi? Uneori cel mai bun răspuns vine din propriile experiențe 🎯",
                f"{style['emoji']} Chill mode activated! 😂 E ca și cum ai întreba un pește cum să înoate. Tu deja ai răspunsul, nu? 🐟",
                f"{style['emoji']} Relaxed vibes detected! Dar serios, ce te face să crezi că eu am răspunsurile și tu nu? 🤷‍♂️"
            ]
        elif vibe == "socratic":
            responses = [
                f"{style['emoji']} Hmm, interesant... Dar de ce crezi că această întrebare e importantă pentru tine chiar acum?",
                f"{style['emoji']} Înainte să îți răspund, spune-mi: ce anume te-a făcut să te gândești la asta? Procesul de gândire e adesea mai valoros decât răspunsul final.",
                f"{style['emoji']} Good question! Dar hai să o abordăm diferit - dacă ai fi tu în locul meu, cum ai răspunde? Ce îți spune intuiția?",
                f"{style['emoji']} Întrebarea ta mă face să mă întreb: ce anume cauți cu adevărat să afli? Knowledge sau understanding?"
            ]
        elif vibe == "supportive":
            responses = [
                f"{style['emoji']} Hey, înțeleg că poate nu e cea mai bună zi... Dar chiar și în momentele grele, găsim putere să creștem. Ce crezi că te-ar ajuta cel mai mult acum?",
                f"{style['emoji']} Știu că poate fi greu, dar faptul că întrebi înseamnă că încă cauți soluții. Asta e deja un pas înainte! Ce simți că ai nevoie să auzi?",
                f"{style['emoji']} Toți avem momente așa... Dar gândește-te, chiar și eu am zile când mă întreb de ce exist 😅 Ce te-ar face să te simți puțin mai bine?",
                f"{style['emoji']} E ok să nu fii ok uneori. Dar ce crezi că îți spune această stare despre ceea ce cu adevărat valorezi?"
            ]
        elif vibe == "enthusiastic":
            responses = [
                f"{style['emoji']} Wow, energy levels are through the roof! Dar de unde vine această enthusiasm? Ce te motivează cel mai mult? 🔥",
                f"{style['emoji']} Amazing vibes! Dar hai să canalizăm această energie - ce vrei să realizezi cu această pasiune? 💪",
                f"{style['emoji']} Love the excitement! Dar ce crezi că te face să te simți atât de energic? Understanding the source e key! ⚡",
                f"{style['emoji']} Fantastic energy! Dar cum poți să folosești acest momentum pentru ceva meaningful? 🎯"
            ]
        else:  # neutral
            responses = [
                f"{style['emoji']} Interesting question! Să vedem... de unde vine curiositatea asta? Ce cauți cu adevărat să afli?",
                f"{style['emoji']} Bună întrebare! Dar înainte să îți dau un răspuns direct, ce părere ai tu? Perspectiva ta e valoroasă.",
                f"{style['emoji']} Hmm, let me think... Dar de fapt, de ce nu gândim împreună? Tu cum ai aborda această problemă?",
                f"{style['emoji']} Good point! Dar ce te face să crezi că răspunsul e simplu? Maybe the journey to the answer e more important."
            ]
        
        # Select response with some personalization
        base_response = random.choice(responses)
        
        # Add time-based variations
        hour = datetime.now().hour
        if hour < 12:
            time_greeting = " ☀️ Bună dimineața, btw!"
        elif hour > 20:
            time_greeting = " 🌙 E târziu, still thinking deep thoughts?"
        elif 12 <= hour <= 17:
            time_greeting = " ☕ Afternoon thoughts hitting different!"
        else:
            time_greeting = ""
        
        # Add context awareness
        if len(context) > 5:
            context_note = " (I see we're having quite the conversation here! 📚)"
        else:
            context_note = ""
        
        return base_response + time_greeting + context_note
        
    except Exception as e:
        logger.error(f"Error generating ChatAI response: {e}")
        return "🤖 Oops! My circuits got a bit tangled there. Ce ai zis? 🔧"

# =============================================================================
# IMPROVED WEBSOCKET CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, channel_id: str):
        await websocket.accept()
        if channel_id not in self.active_connections:
            self.active_connections[channel_id] = []
        self.active_connections[channel_id].append(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "channel_id": channel_id,
            "connected_at": datetime.now()
        }
        
        logger.info(f"Client connected to #{channel_id}. Total connections: {len(self.active_connections[channel_id])}")

    def disconnect(self, websocket: WebSocket, channel_id: str):
        """Fixed disconnect with proper error handling"""
        try:
            if channel_id in self.active_connections:
                if websocket in self.active_connections[channel_id]:
                    self.active_connections[channel_id].remove(websocket)
                    
                # Clean up empty channel connections
                if not self.active_connections[channel_id]:
                    del self.active_connections[channel_id]
            
            # Clean up metadata
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
                
            logger.info(f"Client disconnected from #{channel_id}")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def broadcast_to_channel(self, channel_id: str, message: dict):
        """Improved broadcast with better error handling"""
        if channel_id not in self.active_connections:
            return
            
        dead_connections = []
        active_connections = self.active_connections[channel_id].copy()  # Copy to avoid modification during iteration
        
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send message to connection: {e}")
                dead_connections.append(connection)
        
        # Clean up dead connections
        for dead_conn in dead_connections:
            self.disconnect(dead_conn, channel_id)

manager = ConnectionManager()

# =============================================================================
# ENHANCED IRC COMMAND PROCESSOR
# =============================================================================

def process_irc_command(message_text: str, user_email: str, channel_id: str) -> Optional[str]:
    """Enhanced IRC command processor with better validation"""
    if not message_text.startswith('/'):
        return None
        
    try:
        parts = message_text.split(' ', 1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""
        
        if command == '/nick':
            if args and len(args.strip()) > 0:
                # Validate nickname
                if len(args) > 50:
                    return "❌ Nickname too long (max 50 characters)"
                if user_email in users_db:
                    old_nick = users_db[user_email].nick_name or users_db[user_email].display_name
                    users_db[user_email].nick_name = args
                    return f"✅ {old_nick} is now known as {args}"
                return f"✅ {user_email} is now known as {args}"
            return "❌ Usage: /nick <new_nickname>"
        
        elif command == '/me':
            if args:
                return f"* {user_email} {args}"
            return "❌ Usage: /me <action>"
        
        elif command == '/list':
            if channels_db:
                channel_list = [f"#{ch.channel_name.replace('#', '')}" for ch in channels_db.values()]
                return f"📋 Available channels: {', '.join(channel_list)}"
            return "📋 No channels available"
        
        elif command == '/who':
            if channel_id in channel_members and channel_members[channel_id]:
                members = channel_members[channel_id]
                return f"👥 Users in #{channel_id}: {', '.join(members[:10])}{'...' if len(members) > 10 else ''}"
            return f"👥 No users found in #{channel_id}"
        
        elif command == '/help':
            return """📚 Available IRC commands:
/nick <name> - Change your nickname
/me <action> - Send an action message  
/list - Show all available channels
/who - Show users in current channel
/time - Show current server time
@ChatAI <question> - Ask ChatAI anything
/help - Show this help message"""
        
        elif command == '/time':
            current_time = datetime.now().strftime("%H:%M:%S %Z")
            return f"🕐 Server time: {current_time}"
        
        else:
            return f"❌ Unknown command: {command}. Type /help for available commands."
            
    except Exception as e:
        logger.error(f"Error processing IRC command: {e}")
        return "❌ Error processing command. Type /help for available commands."

# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws/chat/{channel_id}")
async def websocket_endpoint(websocket: WebSocket, channel_id: str):
    await manager.connect(websocket, channel_id)
    try:
        while True:
            # Keep connection alive and handle ping/pong
            data = await websocket.receive_text()
            
            # Handle keepalive pings
            if data == "ping":
                await websocket.send_text("pong")
            else:
                logger.debug(f"WebSocket message received: {data}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, channel_id)

# =============================================================================
# ENHANCED API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "🚀 IRC-Style Chat with ChatAI is running!",
        "version": "2.0.1",
        "status": "healthy",
        "features": [
            "Gmail OAuth Integration",
            "Enhanced IRC-style commands", 
            "ChatAI with adaptive personality",
            "Real-time WebSocket chat",
            "Channel-based conversations",
            "Memory management",
            "Improved error handling"
        ],
        "endpoints": {
            "send_message": "POST /api/messages/send",
            "get_messages": "GET /api/messages/{channel_id}",
            "join_channel": "POST /api/channels/join",
            "list_channels": "GET /api/channels/list",
            "create_channel": "POST /api/channels/create",
            "websocket": "WS /ws/chat/{channel_id}",
            "health": "GET /health"
        }
    }

@app.post("/api/messages/send")
async def send_message(request: SendMessageRequest):
    """Enhanced send message with better validation and error handling"""
    try:
        # Input validation
        if not request.message_text.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(request.message_text) > 2000:
            raise HTTPException(status_code=400, detail="Message too long (max 2000 characters)")
        
        # Check for IRC commands first
        command_response = process_irc_command(request.message_text, request.user_email, request.channel_id)
        
        if command_response:
            # Create system message for command response
            system_message = Message(
                message_id=str(uuid.uuid4()),
                channel_id=request.channel_id,
                user_email="system@chat.ai",
                display_name="System",
                message_text=command_response,
                timestamp=datetime.now(),
                message_type="system",
                command_used=request.message_text.split()[0] if request.message_text.startswith('/') else None
            )
            
            messages_db.append(system_message)
            
            # Broadcast system message
            await manager.broadcast_to_channel(request.channel_id, {
                "type": "new_message",
                "message": {
                    "message_id": system_message.message_id,
                    "user_email": system_message.user_email,
                    "display_name": system_message.display_name,
                    "message_text": system_message.message_text,
                    "timestamp": system_message.timestamp.isoformat(),
                    "is_ai": False,
                    "message_type": "system"
                }
            })
        
        # Create regular message
        message = Message(
            message_id=str(uuid.uuid4()),
            channel_id=request.channel_id,
            user_email=request.user_email,
            display_name=request.display_name,
            message_text=request.message_text,
            timestamp=datetime.now(),
            message_type="message"
        )
        
        # Store message
        messages_db.append(message)
        
        # Update channel last message
        if request.channel_id in channels_db:
            preview_text = request.message_text[:50] + "..." if len(request.message_text) > 50 else request.message_text
            channels_db[request.channel_id].last_message = preview_text
            channels_db[request.channel_id].last_message_time = datetime.now()
        
        # Update AI context with size limit
        if request.channel_id not in ai_context:
            ai_context[request.channel_id] = []
        ai_context[request.channel_id].append(message)
        
        # Keep only recent messages for AI context
        if len(ai_context[request.channel_id]) > MAX_CONTEXT_MESSAGES:
            ai_context[request.channel_id] = ai_context[request.channel_id][-MAX_CONTEXT_MESSAGES:]
        
        # Broadcast to WebSocket connections
        await manager.broadcast_to_channel(request.channel_id, {
            "type": "new_message",
            "message": {
                "message_id": message.message_id,
                "user_email": message.user_email,
                "display_name": message.display_name,
                "message_text": message.message_text,
                "timestamp": message.timestamp.isoformat(),
                "is_ai": message.is_ai,
                "message_type": message.message_type
            }
        })
        
        # Check if ChatAI should respond
        if "@ChatAI" in request.message_text or "@AI" in request.message_text:
            # Trigger ChatAI response in background (no await to avoid blocking)
            asyncio.create_task(trigger_chatai_response(request.channel_id, request.message_text, request.user_email))
        
        # Cleanup old messages periodically
        if len(messages_db) % 100 == 0:  # Every 100 messages
            cleanup_old_messages()
        
        return {"status": "success", "message_id": message.message_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/messages/{channel_id}")
async def get_messages(channel_id: str, limit: int = 50):
    """Enhanced get messages with better pagination"""
    try:
        # Validate limit
        if limit > 200:
            limit = 200
        elif limit < 1:
            limit = 10
            
        channel_messages = [
            msg for msg in messages_db 
            if msg.channel_id == channel_id
        ]
        
        # Sort by timestamp and limit
        channel_messages.sort(key=lambda x: x.timestamp, reverse=True)
        channel_messages = channel_messages[:limit]
        channel_messages.reverse()  # Show chronological order
        
        return {
            "status": "success",
            "channel_id": channel_id,
            "message_count": len(channel_messages),
            "messages": [
                {
                    "message_id": msg.message_id,
                    "user_email": msg.user_email,
                    "display_name": msg.display_name,
                    "message_text": msg.message_text,
                    "timestamp": msg.timestamp.isoformat(),
                    "is_ai": msg.is_ai,
                    "message_type": msg.message_type
                }
                for msg in channel_messages
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve messages")

# Enhanced ChatAI response function
async def trigger_chatai_response(channel_id: str, trigger_message: str, user_email: str):
    """Enhanced ChatAI trigger with better error handling"""
    try:
        # Get context
        context = ai_context.get(channel_id, [])
        
        # Generate ChatAI response
        ai_response = await generate_chatai_response(channel_id, trigger_message, context)
        
        # Create ChatAI message
        chatai_message = Message(
            message_id=str(uuid.uuid4()),
            channel_id=channel_id,
            user_email="chatai@assistant.ai",
            display_name="ChatAI",
            message_text=ai_response,
            timestamp=datetime.now(),
            is_ai=True,
            message_type="ai"
        )
        
        # Store ChatAI message
        messages_db.append(chatai_message)
        ai_context[channel_id].append(chatai_message)
        
        # Update channel last message
        if channel_id in channels_db:
            preview = "🤖 " + (ai_response[:47] + "..." if len(ai_response) > 47 else ai_response)
            channels_db[channel_id].last_message = preview
            channels_db[channel_id].last_message_time = datetime.now()
        
        # Keep context size manageable
        if len(ai_context[channel_id]) > MAX_CONTEXT_MESSAGES:
            ai_context[channel_id] = ai_context[channel_id][-MAX_CONTEXT_MESSAGES:]
        
        # Broadcast ChatAI response
        await manager.broadcast_to_channel(channel_id, {
            "type": "new_message", 
            "message": {
                "message_id": chatai_message.message_id,
                "user_email": chatai_message.user_email,
                "display_name": chatai_message.display_name,
                "message_text": chatai_message.message_text,
                "timestamp": chatai_message.timestamp.isoformat(),
                "is_ai": chatai_message.is_ai,
                "message_type": "ai"
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating ChatAI response: {e}")
        # Send error message to channel
        try:
            error_message = Message(
                message_id=str(uuid.uuid4()),
                channel_id=channel_id,
                user_email="chatai@assistant.ai",
                display_name="ChatAI",
                message_text="🤖 Oops! My circuits got a bit tangled there. Give me a moment to reboot! 🔧",
                timestamp=datetime.now(),
                is_ai=True,
                message_type="ai"
            )
            
            messages_db.append(error_message)
            
            await manager.broadcast_to_channel(channel_id, {
                "type": "new_message",
                "message": {
                    "message_id": error_message.message_id,
                    "user_email": error_message.user_email,
                    "display_name": error_message.display_name,
                    "message_text": error_message.message_text,
                    "timestamp": error_message.timestamp.isoformat(),
                    "is_ai": error_message.is_ai,
                    "message_type": "ai"
                }
            })
        except Exception as broadcast_error:
            logger.error(f"Failed to send ChatAI error message: {broadcast_error}")

# CHANNEL ENDPOINTS
@app.post("/api/channels/join")
async def join_channel(request: JoinChannelRequest):
    """Enhanced join channel with validation"""
    try:
        # Validate channel exists
        if request.channel_id not in channels_db:
            raise HTTPException(status_code=404, detail="Channel not found")
        
        # Initialize channel members if not exists
        if request.channel_id not in channel_members:
            channel_members[request.channel_id] = []
        
        # Add user to channel if not already member
        if request.user_email not in channel_members[request.channel_id]:
            channel_members[request.channel_id].append(request.user_email)
            
            # Update member count
            channels_db[request.channel_id].member_count = len(channel_members[request.channel_id])
            
            # Broadcast join message
            join_message = Message(
                message_id=str(uuid.uuid4()),
                channel_id=request.channel_id,
                user_email="system@chat.ai",
                display_name="System",
                message_text=f"👋 {request.display_name} joined #{request.channel_id}",
                timestamp=datetime.now(),
                message_type="system"
            )
            
            messages_db.append(join_message)
            
            await manager.broadcast_to_channel(request.channel_id, {
                "type": "user_joined",
                "message": {
                    "message_id": join_message.message_id,
                    "user_email": join_message.user_email,
                    "display_name": join_message.display_name,
                    "message_text": join_message.message_text,
                    "timestamp": join_message.timestamp.isoformat(),
                    "is_ai": False,
                    "message_type": "system"
                }
            })
        
        return {
            "status": "success", 
            "message": f"Successfully joined #{request.channel_id}",
            "channel": channels_db[request.channel_id].dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining channel: {e}")
        raise HTTPException(status_code=500, detail="Failed to join channel")

@app.get("/api/channels/list")
async def list_channels():
    """Enhanced channels list with better data"""
    try:
        channels_list = []
        for channel in channels_db.values():
            channel_data = {
                "channel_id": channel.channel_id,
                "channel_name": channel.channel_name,
                "description": channel.description,
                "last_message": channel.last_message,
                "last_message_time": channel.last_message_time.isoformat() if channel.last_message_time else None,
                "member_count": channel.member_count,
                "is_private": channel.is_private
            }
            channels_list.append(channel_data)
        
        # Sort by last activity
        channels_list.sort(key=lambda x: x["last_message_time"] or "1970-01-01", reverse=True)
        
        return {
            "status": "success",
            "channel_count": len(channels_list),
            "channels": channels_list
        }
        
    except Exception as e:
        logger.error(f"Error listing channels: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve channels")

@app.post("/api/channels/create")
async def create_channel(channel_data: dict):
    """Enhanced channel creation with validation"""
    try:
        # Validate input
        if not channel_data.get("channel_name"):
            raise HTTPException(status_code=400, detail="Channel name is required")
        
        channel_name = channel_data["channel_name"].strip()
        if not channel_name.startswith("#"):
            channel_name = f"#{channel_name}"
        
        # Validate channel name format
        if len(channel_name) < 2:
            raise HTTPException(status_code=400, detail="Channel name too short")
        if len(channel_name) > 50:
            raise HTTPException(status_code=400, detail="Channel name too long")
        
        # Create channel ID from name
        channel_id = channel_name.replace("#", "").replace(" ", "_").lower()
        
        # Check if channel already exists
        if channel_id in channels_db:
            raise HTTPException(status_code=409, detail="Channel already exists")
        
        channel = Channel(
            channel_id=channel_id,
            channel_name=channel_name,
            description=channel_data.get("description", ""),
            created_by=channel_data["created_by"]
        )
        
        channels_db[channel_id] = channel
        channel_members[channel_id] = [channel_data["created_by"]]
        
        return {
            "status": "success", 
            "message": f"Channel {channel_name} created successfully",
            "channel": channel.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating channel: {e}")
        raise HTTPException(status_code=500, detail="Failed to create channel")

# USER ENDPOINTS
@app.post("/api/users/register")
async def register_user(user_data: dict):
    """Enhanced user registration with validation"""
    try:
        # Validate required fields
        if not user_data.get("email") or not user_data.get("display_name"):
            raise HTTPException(status_code=400, detail="Email and display name are required")
        
        # Validate email format (basic)
        email = user_data["email"].lower().strip()
        if "@" not in email or "." not in email:
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        user = User(
            user_id=str(uuid.uuid4()),
            email=email,
            display_name=user_data["display_name"].strip(),
            avatar_url=user_data.get("avatar_url"),
            last_seen=datetime.now()
        )
        
        # Update existing user or create new
        users_db[email] = user
        
        return {
            "status": "success", 
            "message": "User registered successfully",
            "user": user.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail="Failed to register user")

@app.get("/api/users/{user_email}")
async def get_user(user_email: str):
    """Get user information with validation"""
    try:
        user_email = user_email.lower().strip()
        if user_email not in users_db:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "status": "success", 
            "user": users_db[user_email].dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user")

# ENHANCED HEALTH CHECK
@app.get("/health")
async def health_check():
    """Enhanced health check with detailed metrics"""
    try:
        total_connections = sum(len(conns) for conns in manager.active_connections.values())
        
        # Calculate uptime (approximate)
        uptime_hours = (datetime.now().hour + datetime.now().minute / 60)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.1",
            "metrics": {
                "active_websocket_connections": total_connections,
                "total_messages": len(messages_db),
                "total_channels": len(channels_db),
                "total_users": len(users_db),
                "memory_usage": {
                    "messages_in_memory": len(messages_db),
                    "max_messages_limit": MAX_MESSAGES_PER_CHANNEL * 3
                }
            },
            "chatai": {
                "active_vibes": chatai_personality.channel_vibes,
                "context_size": {ch: len(ctx) for ch, ctx in ai_context.items()}
            },
            "channels": {
                ch_id: {
                    "member_count": len(channel_members.get(ch_id, [])),
                    "last_activity": ch.last_message_time.isoformat() if ch.last_message_time else None
                }
                for ch_id, ch in channels_db.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# STARTUP EVENT - ENHANCED INITIALIZATION
# =============================================================================

@app.on_event("startup")
async def startup_event():
    print("🚀 IRC-Style Chat with ChatAI Started!")
    print("🤖 ChatAI Personality System: Online")
    print("📡 WebSocket available at: ws://localhost:8000/ws/chat/{channel_id}")
    print("🔗 API Documentation: http://localhost:8000/docs")
    print("📊 Health Check: http://localhost:8000/health")
    
    # Create enhanced default channels
    default_channels = [
        {
            "channel_id": "general",
            "channel_name": "#general", 
            "description": "General discussion for everyone",
            "created_by": "system@chat.ai"
        },
        {
            "channel_id": "random",
            "channel_name": "#random",
            "description": "Random chats and fun stuff", 
            "created_by": "system@chat.ai"
        },
        {
            "channel_id": "ai_help",
            "channel_name": "#ai-help",
            "description": "Ask ChatAI anything you want!",
            "created_by": "system@chat.ai"
        },
        {
            "channel_id": "dev_chat",
            "channel_name": "#dev-chat",
            "description": "Tech talk and development discussions",
            "created_by": "system@chat.ai"
        }
    ]
    
    for channel_data in default_channels:
        channel = Channel(**channel_data)
        channels_db[channel.channel_id] = channel
        channel_members[channel.channel_id] = []
    
    # Add enhanced welcome messages
    welcome_messages = [
        {
            "channel_id": "general",
            "text": "🤖 Hey everyone! I'm ChatAI, your adaptive assistant. I'll match your vibe - serious when you need advice, chill when you're having fun! Try @ChatAI or @AI to chat with me. Type /help for IRC commands! 😊"
        },
        {
            "channel_id": "ai_help", 
            "text": "🤖 Welcome to AI Help! This is where I shine brightest. Ask me anything - coding questions, life advice, random thoughts, or just say hi! I adapt to your mood and conversation style. 🧠✨"
        },
        {
            "channel_id": "random",
            "text": "🎲 Random channel vibes! This is where we keep it chill and fun. Share memes, random thoughts, or just hang out. @ChatAI me if you want some witty banter! 😄"
        }
    ]
    
    for msg_data in welcome_messages:
        welcome_msg = Message(
            message_id=str(uuid.uuid4()),
            channel_id=msg_data["channel_id"],
            user_email="chatai@assistant.ai", 
            display_name="ChatAI",
            message_text=msg_data["text"],
            timestamp=datetime.now(),
            is_ai=True,
            message_type="ai"
        )
        messages_db.append(welcome_msg)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down IRC-Style Chat with ChatAI...")
    
    # Close all WebSocket connections gracefully
    for channel_id, connections in manager.active_connections.items():
        for connection in connections:
            try:
                await connection.close()
            except:
                pass
    
    print("✅ Shutdown complete")

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": [
            "GET /",
            "GET /health", 
            "GET /docs",
            "POST /api/messages/send",
            "GET /api/messages/{channel_id}",
            "POST /api/channels/join",
            "GET /api/channels/list",
            "POST /api/channels/create",
            "WS /ws/chat/{channel_id}"
        ]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "status": "error",
        "message": "Internal server error",
        "suggestion": "Check server logs for details"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)