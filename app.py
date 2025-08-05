from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Mediapipe setup
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/blur', methods=['POST'])
def blur_faces():
    image = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
    image.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return "❌ Couldn't read image."

    height, width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use model_selection=1 for detecting small/distant faces
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3) as detector:
        results = detector.process(img_rgb)

        if not results.detections:
            print("❌ No faces detected.")
        else:
            print(f"✅ Detected {len(results.detections)} face(s)")

            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Add padding to make blur cleaner
                pad = 10
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(w + 2 * pad, width - x)
                h = min(h + 2 * pad, height - y)

                # Apply blur
                face_region = img[y:y+h, x:x+w]
                face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                img[y:y+h, x:x+w] = face_region

    # Save result image
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(result_path, img)

    return render_template('result.html', result_image='result.jpg')

if __name__ == '__main__':
    app.run(debug=True)
