from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO

from config import Config
from camera import Camera
from model import Model
from measurement import MeasurementService
from video_stream import VideoStreamer

# Initialize dependencies
config = Config()
camera = Camera(config)
model = Model(config)
measurement_service = MeasurementService(camera, model)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
video_streamer = VideoStreamer(camera, model, measurement_service, socketio)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    state = measurement_service.get_state()
    return render_template('dashboard.html', calibrated=camera.camera_matrix is not None, latest=state.get("latest"), reference=state.get("reference"), offsets=state.get("offsets"))

@app.route('/video')
def video():
    return Response(video_streamer.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def api_state():
    state = measurement_service.get_state()
    return jsonify({
        **state,
        "calibrated": camera.camera_matrix is not None,
        "focal_length_px": camera.focal_length_px,
        "principal_point": {"x": camera.principal_point[0], "y": camera.principal_point[1]},
        "cartesian": {"x_cm": state["latest"]["cartisian"]["x_cm"] if state["latest"] else None,
                      "z_cm": state["latest"]["cartisian"]["z_cm"] if state["latest"] else None},
    })

@app.route('/api/reference', methods=['POST', 'DELETE'])
def api_reference():
    if request.method == 'POST':
        success, data = measurement_service.set_reference()
        if not success:
            return data, 400
        return "", 200

    success, message = measurement_service.clear_reference()
    if not success:
        return message, 400
    return "", 200

if __name__ == "__main__":
    try:
        socketio.run(app, host="0.0.0.0", port=config.PORT, debug=True, allow_unsafe_werkzeug=True)
    finally:
        camera.release()