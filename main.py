from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd

app = FastAPI()

@app.get("/news", response_class=HTMLResponse)
def get_news():
    # Import the scraping script as a module
    import Financial_News_Scraping_Sentiment_NER as scraper

    # Run the scraping logic
    # The script seems to store the final DataFrame in a variable, but since the code is not fully shown,
    # let's assume the final DataFrame is called 'df' at the end of the script.
    # If not, you may need to adjust this variable name.
    # We'll simulate running the script by calling its main logic if it exists.
    # If the script only runs on import, this will suffice.

    # If the script defines a function to run everything, call it here.
    # Otherwise, if it just runs on import, the DataFrame should be available.

    # Try to get the DataFrame. If not present, return an error.
    df = None
    if hasattr(scraper, 'df'):
        df = scraper.df
    elif hasattr(scraper, 'all_articles_data'):
        # Try to convert the list of dicts to DataFrame
        df = pd.DataFrame(scraper.all_articles_data)
        df.drop(columns = ['url', 'positive_score', 'negative_score', 'neutral_score'], inplace = True)
    else:
        return HTMLResponse(content="<h2>No DataFrame found in the scraper module.</h2>", status_code=500)

    # Convert DataFrame to HTML
    html = df.to_html(index=False, escape=False)
    return HTMLResponse(content=html, status_code=200)
