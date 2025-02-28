from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import Config

app = FastAPI()

# 加载训练好的模型
model = YOLO(Config.model_path, task='detect')

def format_detection(results):
    """格式化检测结果"""
    output = []
    boxes = results[0].boxes
    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        item = {
            "class_id": int(cls),
            "class_name": Config.names[int(cls)],
            "confidence": float(conf),
            "bbox": [int(x) for x in box.tolist()]  # [xmin, ymin, xmax, ymax]
        }
        output.append(item)
    return output

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        # 读取上传的图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 执行检测
        results = model(img)
        
        # 格式化输出
        detections = format_detection(results)
        
        return JSONResponse({
            "status": "success",
            "detections": detections,
            "detected_count": len(detections)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)