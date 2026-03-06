import io
import base64
import datetime
import traceback
from flask import Flask, render_template, request, jsonify, send_file
from dashboard_generator import generate_dashboard

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


@app.errorhandler(Exception)
def handle_exception(e):
    """Catch-all so the client always gets JSON, never an HTML traceback."""
    tb = traceback.format_exc()
    print(f"[ERROR] {e}\n{tb}")
    return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')


def _read_upload(name, required=True):
    """Read an uploaded file fully into bytes; return None for missing optionals."""
    f = request.files.get(name)
    if f is None or f.filename == '':
        if required:
            return name  # sentinel: the *name* of the missing field
        return None
    return f.read()


@app.route('/generate', methods=['POST'])
def generate():
    delivery_bytes = _read_upload('delivery')
    mime_bytes = _read_upload('mime_csv')
    grader_bytes = _read_upload('grader', required=False)
    ocr_bytes = _read_upload('ocr', required=False)
    title = request.form.get('title', '').strip() or None

    missing = [v for v in (delivery_bytes, mime_bytes) if isinstance(v, str)]
    if missing:
        labels = {'delivery': 'Delivery JSONL', 'mime_csv': 'DAMM Tasks CSV'}
        names = [labels.get(m, m) for m in missing]
        return jsonify({'error': f'Missing required files: {", ".join(names)}'}), 400

    try:
        png_bytes = generate_dashboard(
            delivery_bytes=delivery_bytes,
            mime_csv_bytes=mime_bytes,
            grader_bytes=grader_bytes,
            ocr_bytes=ocr_bytes,
            title=title,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[GENERATE ERROR] {e}\n{tb}")
        return jsonify({'error': f'Dashboard generation failed: {e}'}), 500

    b64 = base64.b64encode(png_bytes).decode('utf-8')
    return jsonify({'image': b64})


@app.route('/download', methods=['POST'])
def download():
    delivery_bytes = _read_upload('delivery')
    mime_bytes = _read_upload('mime_csv')
    grader_bytes = _read_upload('grader', required=False)
    ocr_bytes = _read_upload('ocr', required=False)
    title = request.form.get('title', '').strip() or None

    missing = [v for v in (delivery_bytes, mime_bytes) if isinstance(v, str)]
    if missing:
        return jsonify({'error': 'Missing required files'}), 400

    try:
        png_bytes = generate_dashboard(
            delivery_bytes=delivery_bytes,
            mime_csv_bytes=mime_bytes,
            grader_bytes=grader_bytes,
            ocr_bytes=ocr_bytes,
            title=title,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[DOWNLOAD ERROR] {e}\n{tb}")
        return jsonify({'error': f'Dashboard generation failed: {e}'}), 500

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        io.BytesIO(png_bytes),
        mimetype='image/png',
        as_attachment=True,
        download_name=f'dashboard_{timestamp}.png',
    )


if __name__ == '__main__':
    app.run(debug=True, port=5050)
