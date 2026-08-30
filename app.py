from datetime import datetime
from functools import wraps
import os
import uuid
# Flask and extensions for routing, CORS, JWT authentication, and Database ORM
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
# Custom ML predictor module (loads the saved model and runs inference)
from utils.predictor import predict_disease
# Initialize the Flask application
app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization", "Accept"],
    methods=["GET", "POST", "OPTIONS"],
)
# Enable Cross-Origin Resource Sharing (CORS) to allow requests from the React Native app and Web Dashboard

# --- Application Configuration ---
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app_v3.db?timeout=20"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.getenv(
    "JWT_SECRET_KEY",
    "change-this-development-secret-before-production",
)
# Initialize Database and JWT Manager with the Flask app
db = SQLAlchemy(app)
jwt = JWTManager(app)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    mobile = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), default="user", nullable=False)
    predictions = db.relationship("Prediction", backref="owner", lazy=True)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    image_path = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

# Create database tables automatically if they do not exist
with app.app_context():
    db.create_all()

# --- Custom Decorators & Helpers ---
def admin_required(handler):
    @wraps(handler)
    @jwt_required()
    def wrapped(*args, **kwargs):
        user_id = int(get_jwt_identity())
        user = db.session.get(User, user_id)

        if user is None:
            return jsonify({"error": "User not found"}), 404

        if user.role != "user":
            return jsonify({
                "error": "Super Admin access required"
            }), 403

        return handler(*args, **kwargs)

    return wrapped




def prediction_json(prediction, include_user=False):
    result = {
        "id": prediction.id,
        "disease": prediction.disease,
        "confidence": prediction.confidence,
        "latitude": prediction.latitude,
        "longitude": prediction.longitude,
        "timestamp": prediction.timestamp.isoformat(),
    }
    if include_user:
        result["username"] = prediction.owner.username
    return result

# --- API Endpoints ---
@app.get("/")
def home():
    return jsonify({"message": "API is running"})


@app.post("/register")
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    mobile = str(data.get("mobile", "")).strip()
    password = str(data.get("password", ""))

    # Determine user role based on request source (Mobile app vs. Web dashboard)
    requested_role = data.get("role", "user")
    source = data.get("source", "app")

    if source == "web" or requested_role == "user":
        role = "user"
    else:
        role = "user"
    # Validation: Ensure all necessary fields are provided
    if not username or not email or not mobile or not password:
        return jsonify({"error": "Username, email, mobile, and password are required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already exists"}), 400

    try:
        user = User(
            username=username,
            email=email,
            mobile=mobile,
            password=generate_password_hash(password),
            role=role,
        )
        db.session.add(user)
        db.session.commit()
        return jsonify({"message": "User created successfully", "role": user.role}), 201
    except Exception as error:
        db.session.rollback()
        return jsonify({"error": str(error)}), 500


@app.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    user = User.query.filter_by(username=username).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid credentials"}), 401
    if data.get("source", "app") == "web" and user.role != "user":
        return jsonify({"error": "Only Super Admins can log in via the web app"}), 403

    return jsonify({
        "access_token": create_access_token(identity=str(user.id)),
        "role": user.role,
        "username": user.username,
    })


@app.post("/predict")
@jwt_required()
def predict():
    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify({"error": "An image file is required"}), 400

    try:
        safe_name = secure_filename(image.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{safe_name}")
        image.save(filepath)
        disease, confidence = predict_disease(filepath)
        prediction = Prediction(
            disease=str(disease),
            confidence=float(confidence),
            latitude=request.form.get("latitude", type=float),
            longitude=request.form.get("longitude", type=float),
            image_path=filepath,
            user_id=int(get_jwt_identity()),
        )
        db.session.add(prediction)
        db.session.commit()
        return jsonify({"disease": disease, "confidence": confidence, "message": "Analysis complete and saved"})
    except Exception as error:
        db.session.rollback()
        return jsonify({"error": str(error)}), 500


@app.get("/history")
@jwt_required()
def history():
    user_id = int(get_jwt_identity())
    records = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.timestamp.desc()).all()
    return jsonify({"history": [prediction_json(record) for record in records]})



@app.route("/admin/history", methods=["GET"])
@admin_required
def admin_history():
    records = Prediction.query.order_by(
        Prediction.timestamp.desc()
    ).all()

    return jsonify({
        "history": [
            prediction_json(record, include_user=True)
            for record in records
        ]
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
