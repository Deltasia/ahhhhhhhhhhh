from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO
import signal
import sys

from config import Config
from camera import Camera, enumerate_cameras, clear_camera_cache
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
video_streamer = VideoStreamer(camera, model, measurement_service, socketio, config)

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

@app.route('/api/cameras')
def api_cameras():
    """Get list of available cameras."""
    cameras = enumerate_cameras()
    current_camera = getattr(camera, 'camera_index', 0)
    return jsonify({
        'cameras': cameras,
        'current': current_camera
    })

@app.route('/api/camera/switch', methods=['POST'])
def api_switch_camera():
    """Switch to a different camera."""
    data = request.get_json()
    if not data or 'index' not in data:
        return jsonify({'error': 'Camera index required'}), 400
    
    try:
        camera_index = int(data['index'])
        success = camera.switch_camera(camera_index)
        if success:
            # Clear cache after successful switch so next enumeration is fresh
            clear_camera_cache()
            return jsonify({'message': f'Switched to camera {camera_index}'}), 200
        else:
            return jsonify({'error': f'Failed to switch to camera {camera_index}'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid camera index'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cameras/refresh', methods=['POST'])
def api_refresh_cameras():
    """Force refresh of camera list by clearing cache."""
    clear_camera_cache()
    cameras = enumerate_cameras(use_cache=False)
    current_camera = getattr(camera, 'camera_index', 0)
    return jsonify({
        'cameras': cameras,
        'current': current_camera,
        'message': 'Camera list refreshed'
    })

def signal_handler(sig, frame):
    """Handle shutdown signals for proper cleanup."""
    print(f"\nReceived signal {sig}, shutting down gracefully...")
    camera.release()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        socketio.run(app, host="0.0.0.0", port=config.PORT, debug=True, allow_unsafe_werkzeug=True)
    finally:
        camera.release()