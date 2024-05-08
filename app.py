from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Route to load the Boston Housing dataset

# Route for your main page
@app.route('/')
def index():
    return render_template('main.html')

# Route for the index page
@app.route('/index')
def index_page():
    return render_template('index.html')


# Route for the multiple regression page
@app.route('/multiple')
def multiple():
    return render_template('multiple.html')


@app.route('/mat')
def mat():
    return render_template('mat.html')

if __name__ == '__main__':
    app.run(debug=True)