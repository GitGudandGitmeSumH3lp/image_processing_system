CREATE TABLE IF NOT EXISTS ocr_results (
    id SERIAL PRIMARY KEY,
    text_content TEXT,
    image_path TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    predicted_destination TEXT,
    match_score FLOAT
);
