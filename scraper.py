import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Function to scrape news articles from the New York Times website
def scrape_news_articles(url):
    # Send GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all articles
        articles = soup.find_all('article', class_='story-body')

        # Extract article data
        article_data = []
        for article in articles:
            # Get article title
            title = article.find('h2', class_='story-title').text.strip()

            # Get article section
            section = article.find('a', class_='story-link').text.strip()

            # Get article summary
            summary = article.find('p', class_='summary').text.strip()

            # Combine article data into a dictionary
            article_info = {
                'title': title,
                'section': section,
                'summary': summary
            }

            # Add article data to the list
            article_data.append(article_info)

        # Return the list of article data
        return article_data
    else:
        print('Error: Unable to scrape news articles from {}'.format(url))
        return None

# Scrape news articles from the New York Times website
news_articles = scrape_news_articles('https://www.nytimes.com/section/world/latest')

# Create a pandas DataFrame from the scraped data
df = pd.DataFrame(news_articles)

# Split the data into training and testing sets
X = df['summary']
y = df['section']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = classifier.predict(X_test)
accuracy = classifier.score(X_test, y_test)
print('Accuracy:', accuracy)

# Create a CSV report of the test evaluation
df_test_eval = pd.DataFrame({
    'Actual Section': y_test,
    'Predicted Section': y_pred
})
df_test_eval.to_csv('test_evaluation.csv', index=False)
