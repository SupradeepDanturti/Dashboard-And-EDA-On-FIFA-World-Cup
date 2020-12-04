"""Routes for parent Flask app."""
from flask import render_template
from flask import current_app as app


@app.route('/')
def home():
    """Landing page."""
    return render_template(
        'index.jinja2',
        title='A simple dashboard on FIFA Dataset',
        description='Click here to view the dashboard.',
        template='home-template',
        body="This is a homepage served with Flask."
    )
