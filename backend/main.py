
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from process import process_pdf, process_image
app = FastAPI()

# Enable CORS for all origins (for both REST API and WebSockets)
origins = [
    "http://localhost:5173",  # Your React app's frontend URL
    "http://localhost:3000",  # If you use another port, you can add that too
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows CORS from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# WebSocket endpoint to process the file
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        json_file_path = "invoice_result.json"
        with open(json_file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        # Receive the file from the frontend (simulated by the uploaded file path here)
        file_name = await websocket.receive_text() 
        uploaded_file = await websocket.receive_bytes()  # Get file in bytes
        

        # # Save the uploaded file
        #  # Generate a unique file name
        # file_path = os.path.join("uploads", file_name)
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # with open(file_path, "wb") as f:
        #     f.write(uploaded_file)

        # Send the file format
        file_extension = file_name.split('.')[-1].lower()
        await websocket.send_text(json.dumps({"message" : f"File format: {file_extension.upper()}", "type" : "success", "finished" : True} ))

        # Send the file size
        # file_size = os.path.getsize(file_path)
        # await websocket.send_text(f"File size: {file_size} bytes")

        if file_extension == "pdf":
            await websocket.send_text("📄 **PDF file detected**")
            await websocket.send_json({"result" : {"text" : file_data} })
            # processing_result = await process_pdf(uploaded_file, websocket, file_name)
            
        else:
            #  await process_image(uploaded_file, websocket, file_name)
            await websocket.send_text("📄 **Image detected**")
            await websocket.send_json({"result" : {"text" : file_data} })
        # Simulate processing completion
        await websocket.send_text(json.dumps({"message" : "File uploaded and processed successfully!", "type" : "success"}))
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(e)
