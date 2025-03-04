from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)

# Configuração do banco de dados SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///calibration.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo do Banco de Dados
class Calibration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    side = db.Column(db.String(20), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    point1_x = db.Column(db.Float, nullable=False)
    point1_y = db.Column(db.Float, nullable=False)
    point2_x = db.Column(db.Float, nullable=False)
    point2_y = db.Column(db.Float, nullable=False)
    point3_x = db.Column(db.Float, nullable=False)
    point3_y = db.Column(db.Float, nullable=False)
    point4_x = db.Column(db.Float, nullable=False)
    point4_y = db.Column(db.Float, nullable=False)

# Criar o banco de dados e as tabelas
with app.app_context():
    db.create_all()
