# AI-Powered URL Shortener

A modern URL shortening service built with Django that includes advanced analytics and AI-powered insights.

## Features

- **URL Shortening**: Create short, easy-to-share links from long URLs
- **Advanced Analytics**: Track clicks, devices, browsers, and operating systems
- **Interactive Dashboards**: Visualize your URL performance with interactive charts
- **AI Insights**: Get intelligent insights about your URL usage patterns
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## Tech Stack

- **Backend**: Django 5.1
- **Frontend**: Bootstrap 5, jQuery, Plotly.js
- **Data Visualization**: Django Plotly Dash, Plotly
- **AI/ML**: LangChain, OpenAI
- **Database**: SQLite (default), compatible with PostgreSQL for production

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/url-shortener.git
cd url-shortener
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file from the example:
```bash
cp .env.example .env
```

5. Edit the `.env` file and add your OpenAI API key for AI insights:
```
OPENAI_API_KEY=your-api-key-here
```

6. Run migrations:
```bash
python manage.py migrate
```

7. Create a superuser (admin):
```bash
python manage.py createsuperuser
```

8. Run the development server:
```bash
python manage.py runserver
```

9. Visit http://127.0.0.1:8000/ in your browser

## Usage

1. Enter a long URL in the input field on the homepage
2. Click "Shorten URL" to generate a short link
3. Copy and share the shortened URL
4. Track analytics for your links in the dashboard
5. View detailed statistics for each URL

## AI Features

This URL shortener includes AI-powered features:

- **Traffic Pattern Analysis**: AI identifies patterns in click behavior
- **User Demographics Insights**: Get insights about your audience based on device and browser data
- **Optimization Recommendations**: Receive suggestions for improving your link performance
- **Anomaly Detection**: Identify unusual patterns or outliers in your URL usage

## Project Structure

```
url_shortener/
├── analytics/            # Analytics app for tracking and visualizing data
├── shortener/            # Core URL shortening functionality
├── templates/            # HTML templates
│   ├── analytics/        # Templates for analytics pages
│   ├── shortener/        # Templates for URL shortening pages
│   └── base.html         # Base template with common elements
├── url_shortener/        # Project settings
├── .env                  # Environment variables (create from .env.example)
├── .env.example          # Example environment variables
├── manage.py             # Django management script
└── requirements.txt      # Project dependencies
```


